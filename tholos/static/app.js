/* Tholos UI helpers: toasts, grid keyboard editing, form reset after htmx posts. */
(function () {
  "use strict";

  function showToast(message) {
    if (!message) return;
    var box = document.getElementById("toasts");
    if (!box) return;
    var toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = typeof message === "string" ? message : message.message || "";
    box.appendChild(toast);
    window.setTimeout(function () {
      toast.remove();
    }, 4000);
  }

  document.body.addEventListener("toast", function (event) {
    showToast(event.detail && event.detail.value !== undefined ? event.detail.value : event.detail);
  });

  document.body.addEventListener("htmx:afterRequest", function (event) {
    var form = event.detail && event.detail.elt;
    if (form && form.tagName === "FORM" && form.hasAttribute("data-reset")) {
      form.reset();
    }
  });

  document.addEventListener("keydown", function (event) {
    var input = event.target;
    if (!input.classList || !input.classList.contains("cell")) return;
    if (event.key === "Enter") {
      event.preventDefault();
      input.form.requestSubmit();
    } else if (event.key === "Escape") {
      input.value = input.defaultValue;
      input.blur();
    }
  });

  document.addEventListener("dblclick", function (event) {
    var input = event.target;
    if (input.classList && input.classList.contains("cell")) {
      input.focus();
      input.select();
    }
  });
})();
