// Right-side inspector drawer with backdrop and ESC close.

let onCloseFn = null;

export function openDrawer(contentEl, { onClose } = {}) {
  const drawer = document.getElementById("drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  drawer.innerHTML = "";
  drawer.appendChild(contentEl);
  drawer.classList.add("open");
  backdrop.classList.add("open");
  onCloseFn = onClose ?? null;
}

export function closeDrawer() {
  const drawer = document.getElementById("drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  drawer.classList.remove("open");
  backdrop.classList.remove("open");
  if (onCloseFn) {
    const fn = onCloseFn;
    onCloseFn = null;
    fn();
  }
}

document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDrawer();
});
