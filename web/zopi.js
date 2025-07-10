import { app } from '../../scripts/app.js'
import { ComfyWidgets } from '../../scripts/widgets.js';

app.registerExtension({
  name: "zopi/LoadTensortRTAndCheckpoint",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "LoadTensortRTAndCheckpoint") return;

    function showTexts(texts) {
      if (this.widgets) {
        for (let i = 1; i < this.widgets.length; i++) this.widgets[i].onRemove?.();
        this.widgets.length = 1;
      }
      const LABELS = ["SD", "Type"];
      let l = 0;
      for (const value of texts) {
        const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: false }], app).widget;
        w.disabled = true;
        w.name = LABELS[l++] ?? '?';
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
        showTexts.call(this, this.widgets_values.slice(1 /* native input widget count */));
      }
    };
  },
})
