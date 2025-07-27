import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "zopi/LoadTensortRTAndCheckpoint",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "LoadTensortRTAndCheckpoint") return;

    function showTexts(texts) {
      if (this.widgets) {
        for (let i = 1; i < this.widgets.length; i++)
          this.widgets[i].onRemove?.();
        this.widgets.length = 1;
      }
      const LABELS = ["SD", "Type"];
      let l = 0;
      for (const value of texts) {
        const w = ComfyWidgets["STRING"](
          this,
          "text",
          ["STRING", { multiline: false }],
          app
        ).widget;
        w.disabled = true;
        w.name = LABELS[l++] ?? "?";
        w.value = value;
      }
      requestAnimationFrame(() => {
        const sz = this.computeSize();
        if (sz[0] < this.size[0]) {
          sz[0] = this.size[0];
        }
        if (sz[1] < this.size[1]) {
          sz[1] = this.size[1];
        }
        this.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
      });
    }

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      onExecuted?.apply(this, arguments);
      showTexts.call(this, msg.texts);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      if (this.widgets_values?.length) {
        showTexts.call(
          this,
          this.widgets_values.slice(1 /* native input widget count */)
        );
      }
    };
  },
});

const DEBOUNCE_FN_TO_PROMISE = new WeakMap();
const IoDirection = { INPUT: 0, OUTPUT: 1 };
function debounce(fn, ms = 64) {
  if (!DEBOUNCE_FN_TO_PROMISE.get(fn)) {
    DEBOUNCE_FN_TO_PROMISE.set(
      fn,
      wait(ms).then(() => {
        DEBOUNCE_FN_TO_PROMISE.delete(fn);
        fn();
      })
    );
  }
  return DEBOUNCE_FN_TO_PROMISE.get(fn);
}
function wait(ms = 16) {
  if (ms === 16) {
    return new Promise((resolve) => {
      requestAnimationFrame(() => {
        resolve();
      });
    });
  }
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, ms);
  });
}
function getTypeFromSlot(slot, dir, skipSelf = false) {
  let graph = app.graph;
  let type = slot?.type;
  if (!skipSelf && type != null && type != "*") {
    return { type: type, label: slot?.label, name: slot?.name };
  }
  const links = getSlotLinks(slot);
  for (const link of links) {
    const connectedId =
      dir == IoDirection.OUTPUT ? link.link.target_id : link.link.origin_id;
    const connectedSlotNum =
      dir == IoDirection.OUTPUT ? link.link.target_slot : link.link.origin_slot;
    const connectedNode = graph.getNodeById(connectedId);
    // Reversed since if we're traveling down the output we want the connected node's input, etc.
    const connectedSlots =
      dir === IoDirection.OUTPUT ? connectedNode.inputs : connectedNode.outputs;
    let connectedSlot = connectedSlots[connectedSlotNum];
    if (connectedSlot?.type != null && connectedSlot?.type != "*") {
      return {
        type: connectedSlot.type,
        label: connectedSlot?.label,
        name: connectedSlot?.name,
      };
    } else if (connectedSlot?.type == "*") {
      return followConnectionUntilType(connectedNode, dir);
    }
  }
  return null;
}
function followConnectionUntilType(node, dir, slotNum, skipSelf = false) {
  const slots = dir === IoDirection.OUTPUT ? node.outputs : node.inputs;
  if (!slots || !slots.length) {
    return null;
  }
  let type = null;
  if (slotNum) {
    if (!slots[slotNum]) {
      return null;
    }
    type = getTypeFromSlot(slots[slotNum], dir, skipSelf);
  } else {
    for (const slot of slots) {
      type = getTypeFromSlot(slot, dir, skipSelf);
      if (type) {
        break;
      }
    }
  }
  return type;
}
function getSlotLinks(inputOrOutput) {
  const links = [];
  if (!inputOrOutput) {
    return links;
  }
  if (inputOrOutput.links?.length) {
    const output = inputOrOutput;
    for (const linkId of output.links || []) {
      const link = app.graph.links[linkId];
      if (link) {
        links.push({ id: linkId, link: link });
      }
    }
  }
  if (inputOrOutput.link) {
    const input = inputOrOutput;
    const link = app.graph.links[input.link];
    if (link) {
      links.push({ id: input.link, link: link });
    }
  }
  return links;
}

app.registerExtension({
  name: "zopi/EvalPython",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== "EvalPython") return;
    const ALREADY_INPUTS = 1;
    const FREE_INPUTS = 1;
    const ALREADY_OUTPUT = 0;
    const that = nodeType.prototype;

    that.stabilize = function () {
      this.managePorts();
    };
    that.scheduleStabilize = function (ms = 64) {
      return debounce(this.stabilize.bind(this), ms);
    };

    that.numFreeTail = function (what, skip = 0) {
      const ports = what.map((e, i) => [e, i]).slice(skip);
      if (ports.length === 0) return 0;
      const lastUsed = ports
        .map(([p, i]) => (p?.link ?? p.links?.length ? i : -1))
        .reduce((a, b) => Math.max(a, b), -1);
      if (lastUsed === -1) return ports.length;
      return ports.length - lastUsed;
    };
    that.numFreeTailInputs = function () {
      return this.numFreeTail(this.inputs, ALREADY_INPUTS);
    };
    that.managePorts = function () {
      while (
        this.numFreeTailInputs() > FREE_INPUTS &&
        !this.inputs[this.inputs.length - 1]?.link
      ) {
        this.removeInput(this.inputs.length - 1);
      }
      while (this.numFreeTailInputs() < FREE_INPUTS) {
        this.addInput(
          `input[${String(this.inputs.length - 1)}]`,
          "*"
        );
      }
      this.inputs.map((inp, i) => [inp, i]).slice(ALREADY_INPUTS).forEach(([inp, i]) => {
        inp.type = followConnectionUntilType(this, IoDirection.INPUT, i, true)?.type || "*";
        const type = inp.type === "*" ? "" : ` (${inp.type})`;
        inp.label = `${inp.name}${type}`;
      });
      fetch("/zopi/eval_python/types", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expression: this.widgets[0].inputEl.value,
        }),
      })
        .then((res) => res.json())
        .then((json) => {
          for (
            let j = 0, i = ALREADY_OUTPUT;
            j < json.outputs.length;
            ++i, ++j
          ) {
            let [name, typ] = json.outputs[j];
            typ = typ || "*";
            if (i >= this.outputs.length) {
              this.addOutput(name, typ);
            } else {
              this.outputs[i].label = `${name} (${typ})`;
              this.outputs[i].type = typ;
            }
          }
          for (
            let i = this.outputs.length - 1;
            i >= ALREADY_OUTPUT + json.outputs.length;
            --i
          ) {
            this.removeOutput(i);
          }
        });
    };
    const onNodeCreated = that.onNodeCreated;
    that.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      this.widgets[0].inputEl.addEventListener("input", (e) => {
        debounce(this.stabilize.bind(this), 1000);
      });
    };
    const onAdded = that.onAdded;
    that.onAdded = function () {
      onAdded?.apply(this, arguments);
      this.scheduleStabilize();
    };
    const onConnectionsChange = that.onConnectionsChange;
    that.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      this.scheduleStabilize();
    };
  },
});
