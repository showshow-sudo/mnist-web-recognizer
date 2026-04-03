/**
 * 手写画布：高分辨率位图 + CSS 显示尺寸，保证笔迹清晰；
 * 导出 PNG 供服务端做「包围盒 + 居中 + MNIST 式缩放」预处理。
 */
(function () {
  var canvas = document.getElementById("drawCanvas");
  if (!canvas || !canvas.getContext) return;

  var ctx = canvas.getContext("2d");
  var modelSelect = document.getElementById("modelSelect");
  var btnClear = document.getElementById("btnClear");
  var btnPredict = document.getElementById("btnPredict");
  var statusEl = document.getElementById("status");
  var digitOut = document.getElementById("digitOut");
  var confOut = document.getElementById("confOut");
  var probBars = document.getElementById("probBars");

  var drawing = false;
  var lastX = 0;
  var lastY = 0;

  /** 逻辑分辨率（物理像素）；显示宽度由 CSS 控制为 280px */
  var DISPLAY_SIZE = 280;
  var dpr = Math.min(2, window.devicePixelRatio || 1);
  var lineWidthLogical = 20;

  function setStatus(cls, text) {
    statusEl.className = "status " + cls;
    statusEl.textContent = text;
  }

  /** 按 DPR 设置 canvas 内部尺寸并缩放坐标系，使绘制单位 = 逻辑像素 */
  function setupHiDpiCanvas() {
    var w = Math.round(DISPLAY_SIZE * dpr);
    var h = Math.round(DISPLAY_SIZE * dpr);
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = DISPLAY_SIZE + "px";
    canvas.style.height = DISPLAY_SIZE + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = lineWidthLogical;
  }

  function initCanvas() {
    setupHiDpiCanvas();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = lineWidthLogical;
  }

  function getPos(e) {
    var r = canvas.getBoundingClientRect();
    var clientX = e.clientX;
    var clientY = e.clientY;
    if (e.touches && e.touches[0]) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }
    var sx = DISPLAY_SIZE / r.width;
    var sy = DISPLAY_SIZE / r.height;
    return {
      x: (clientX - r.left) * sx,
      y: (clientY - r.top) * sy,
    };
  }

  function startDraw(e) {
    e.preventDefault();
    drawing = true;
    var p = getPos(e);
    lastX = p.x;
    lastY = p.y;
  }

  function moveDraw(e) {
    if (!drawing) return;
    e.preventDefault();
    var p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    lastX = p.x;
    lastY = p.y;
  }

  function endDraw(e) {
    if (e) e.preventDefault();
    drawing = false;
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  canvas.addEventListener("mouseup", endDraw);
  canvas.addEventListener("mouseleave", endDraw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", moveDraw, { passive: false });
  canvas.addEventListener("touchend", endDraw);
  canvas.addEventListener("touchcancel", endDraw);

  btnClear.addEventListener("click", function () {
    initCanvas();
    digitOut.textContent = "—";
    confOut.textContent = "—";
    probBars.innerHTML = "";
    setStatus("idle", "已清空，请重新书写");
  });

  function renderProbs(probs) {
    probBars.innerHTML = "";
    if (!probs || !probs.length) return;
    var maxP = Math.max.apply(null, probs);
    for (var i = 0; i < probs.length; i++) {
      var row = document.createElement("div");
      row.className = "prob-row";
      var pct = maxP > 0 ? (probs[i] / maxP) * 100 : 0;
      row.innerHTML =
        '<span>' +
        i +
        '</span><div class="prob-bar"><i style="width:' +
        pct +
        '%"></i></div><span>' +
        probs[i] +
        "</span>";
      probBars.appendChild(row);
    }
  }

  btnPredict.addEventListener("click", function () {
    var dataUrl = canvas.toDataURL("image/png");
    var model = modelSelect.value;

    setStatus("loading", "识别中…");
    digitOut.textContent = "…";
    confOut.textContent = "…";
    probBars.innerHTML = "";

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model, image: dataUrl }),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          return { ok: res.ok, status: res.status, body: body };
        });
      })
      .then(function (r) {
        if (!r.ok) {
          setStatus("err", r.body.error || "请求失败 (" + r.status + ")");
          digitOut.textContent = "—";
          confOut.textContent = "—";
          return;
        }
        setStatus("ok", "识别成功");
        digitOut.textContent = String(r.body.digit);
        confOut.textContent = String(r.body.confidence);
        renderProbs(r.body.probabilities);
      })
      .catch(function (err) {
        setStatus("err", "网络错误: " + err.message);
        digitOut.textContent = "—";
        confOut.textContent = "—";
      });
  });

  initCanvas();
  setStatus("idle", "等待识别");
})();
