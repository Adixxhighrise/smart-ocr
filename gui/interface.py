"""
gui/interface.py - Smart OCR Desktop Application GUI (dark theme)
v3: PDF support, accuracy display, full pipeline integration,
    multi-page PDF navigation.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import get_logger, load_image, save_text

logger = get_logger("gui")

T = config.THEME


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lbl(parent, text, font=None, fg=None, bg=None, **kw):
    return tk.Label(
        parent, text=text,
        font=font or config.FONT_LABEL,
        fg=fg   or T["text"],
        bg=bg   or T["bg"],
        **kw,
    )


def _frm(parent, bg=None, **kw):
    return tk.Frame(parent, bg=bg or T["bg"], **kw)


def _btn(parent, text, command, variant="primary", **kw):
    colours = {
        "primary": (T["accent"],   T["text"], T["btn_hover"]),
        "success": (T["success"],  "#000",    "#2ecc71"),
        "danger":  (T["error"],    T["text"], "#c0392b"),
        "neutral": (T["bg_card"],  T["text"], T["border"]),
        "pdf":     ("#7c3aed",     "#fff",    "#6d28d9"),
    }
    bg, fg, abg = colours.get(variant, colours["primary"])
    b = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
        font=config.FONT_BUTTON,
        relief="flat", bd=0, cursor="hand2",
        padx=14, pady=8, **kw,
    )
    return b


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PANEL
# ─────────────────────────────────────────────────────────────────────────────

class ImagePanel:
    def __init__(self, parent, title: str, width: int = 420, height: int = 300):
        self._w     = width
        self._h     = height
        self._photo = None

        self._frame = tk.Frame(parent, bg=T["bg_card"], bd=0, relief="flat")

        title_bar = tk.Frame(self._frame, bg=T["bg_secondary"])
        title_bar.pack(fill="x")
        tk.Label(
            title_bar, text=f"  {title}",
            font=config.FONT_HEADING,
            bg=T["bg_secondary"], fg=T["text"],
        ).pack(side="left", pady=5)

        self.canvas = tk.Canvas(
            self._frame,
            width=width, height=height,
            bg=T["bg_panel"], bd=0, highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self._draw_placeholder()

    def grid(self, **kw):  self._frame.grid(**kw)
    def pack(self, **kw):  self._frame.pack(**kw)
    def place(self, **kw): self._frame.place(**kw)

    def set_image(self, pil_image: Image.Image):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()  or self._w
        ch = self.canvas.winfo_height() or self._h

        img = pil_image.copy()
        resample = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )
        img.thumbnail((cw, ch), resample)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = (cw - img.width)  // 2
        y = (ch - img.height) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self._photo)

    def clear(self):
        self._photo = None
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("all")
        cx, cy = self._w // 2, self._h // 2
        self.canvas.create_rectangle(
            cx - 36, cy - 36, cx + 36, cy + 36,
            outline=T["text_muted"], width=1, dash=(4, 4),
        )
        self.canvas.create_text(
            cx, cy, text="No image",
            fill=T["text_muted"], font=config.FONT_LABEL_SM,
        )


# ─────────────────────────────────────────────────────────────────────────────
# BADGE
# ─────────────────────────────────────────────────────────────────────────────

class Badge(tk.Label):
    def __init__(self, parent, text="", color=None, **kw):
        color = color or T["bg_card"]
        super().__init__(
            parent, text=text,
            bg=color, fg="#fff",
            font=config.FONT_LABEL_SM,
            padx=8, pady=3, relief="flat", **kw,
        )

    def set(self, text, color=None):
        self.configure(text=text)
        if color:
            self.configure(bg=color)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class SmartOCRApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(config.APP_TITLE)
        self.geometry(f"{config.APP_WIDTH}x{config.APP_HEIGHT}")
        self.minsize(config.APP_MIN_W, config.APP_MIN_H)
        self.configure(bg=T["bg"])
        self._configure_styles()

        # state
        self._image_path:     str   = ""
        self._is_pdf:         bool  = False
        self._pdf_pages:      list  = []   # list of (page_idx, PIL.Image)
        self._pdf_page_idx:   int   = 0
        self._original_image        = None
        self._processed_image       = None
        self._extracted_text: str   = ""
        self._doc_type:       str   = ""
        self._doc_confidence: float = 0.0
        self._busy:           bool  = False
        self._accuracy:       float = 0.0

        self._build_ui()
        self._refresh_buttons()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ── STYLES ────────────────────────────────────────────────────────────────

    def _configure_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(
            "TProgressbar",
            troughcolor=T["bg_card"],
            background=T["accent"],
            darkcolor=T["accent"],
            lightcolor=T["accent"],
        )

    # ── BUILD UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        row = _frm(self)
        row.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        row.columnconfigure(0, weight=3)
        row.columnconfigure(1, weight=2)
        row.rowconfigure(0, weight=1)
        self._build_left(row)
        self._build_right(row)
        self._build_statusbar()

    # ── HEADER ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = _frm(self, bg=T["bg_secondary"])
        hdr.pack(fill="x")

        left = _frm(hdr, bg=T["bg_secondary"])
        left.pack(side="left", padx=16, pady=10)

        tk.Label(
            left, text="⚙",
            font=(config.FONT_FAMILY, 22),
            bg=T["bg_secondary"], fg=T["accent"],
        ).pack(side="left", padx=(0, 8))

        tk.Label(
            left, text="Smart OCR",
            font=(config.FONT_FAMILY, 15, "bold"),
            bg=T["bg_secondary"], fg=T["text"],
        ).pack(side="left")

        tk.Label(
            left, text="  ·  Handwriting, Print & PDF Recognition",
            font=config.FONT_LABEL,
            bg=T["bg_secondary"], fg=T["text_dim"],
        ).pack(side="left")

        right = _frm(hdr, bg=T["bg_secondary"])
        right.pack(side="right", padx=16)

        # Accuracy badge
        tk.Label(
            right, text="Accuracy:",
            font=config.FONT_LABEL_SM,
            bg=T["bg_secondary"], fg=T["text_dim"],
        ).pack(side="left", padx=(0, 4))

        self._acc_badge = Badge(right, text="—", color=T["bg_card"])
        self._acc_badge.pack(side="left", padx=(0, 12))

        tk.Label(
            right, text="Document type:",
            font=config.FONT_LABEL_SM,
            bg=T["bg_secondary"], fg=T["text_dim"],
        ).pack(side="left", padx=(0, 5))

        self._doc_badge = Badge(right, text="—")
        self._doc_badge.pack(side="left")

        self._conf_lbl = tk.Label(
            right, text="",
            font=config.FONT_LABEL_SM,
            bg=T["bg_secondary"], fg=T["text_dim"],
        )
        self._conf_lbl.pack(side="left", padx=(6, 0))

    # ── LEFT PANEL ────────────────────────────────────────────────────────────

    def _build_left(self, parent):
        left = _frm(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        # Button row
        btns = _frm(left)
        btns.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        self._btn_upload  = _btn(btns, "📂  Upload Image",   self._on_upload,  "primary")
        self._btn_pdf     = _btn(btns, "📄  Upload PDF",     self._on_upload_pdf, "pdf")
        self._btn_process = _btn(btns, "⚙  Process",        self._on_process, "neutral")
        self._btn_extract = _btn(btns, "🔍  Extract Text",   self._on_extract, "neutral")
        self._btn_save    = _btn(btns, "💾  Save Text",      self._on_save,    "success")
        self._btn_clear   = _btn(btns, "✖  Clear",          self._on_clear,   "danger")

        for b in (self._btn_upload, self._btn_pdf, self._btn_process,
                  self._btn_extract, self._btn_save, self._btn_clear):
            b.pack(side="left", padx=(0, 6))

        # Image panels
        self._panel_orig = ImagePanel(left, "Original Image",  width=420, height=300)
        self._panel_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(0, 6))

        self._panel_proc = ImagePanel(left, "Processed Image", width=420, height=300)
        self._panel_proc.grid(row=1, column=1, sticky="nsew", pady=(0, 6))

        # PDF page navigation
        self._pdf_nav = _frm(left)
        self._pdf_nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self._btn_prev_page = _btn(self._pdf_nav, "◀ Prev", self._on_prev_page, "neutral")
        self._btn_prev_page.pack(side="left", padx=(0, 6))

        self._page_var = tk.StringVar(value="")
        tk.Label(
            self._pdf_nav, textvariable=self._page_var,
            font=config.FONT_LABEL_SM, bg=T["bg"], fg=T["text_dim"],
        ).pack(side="left")

        self._btn_next_page = _btn(self._pdf_nav, "Next ▶", self._on_next_page, "neutral")
        self._btn_next_page.pack(side="left", padx=(6, 0))

        self._pdf_nav.grid_remove()  # hidden until PDF loaded

        # Progress bar
        self._progress = ttk.Progressbar(
            left, orient="horizontal", mode="indeterminate",
        )
        self._progress.grid(row=3, column=0, columnspan=2, sticky="ew")

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        right = _frm(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=3)
        right.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)

        # Header row with copy button
        hdr_row = _frm(right)
        hdr_row.grid(row=0, column=0, sticky="ew", pady=(6, 3))
        _lbl(hdr_row, "Extracted Text", font=config.FONT_HEADING).pack(side="left")
        _btn(hdr_row, "⧉ Copy", self._on_copy_text, "neutral").pack(side="right")

        self._txt_out = scrolledtext.ScrolledText(
            right,
            bg=T["bg_panel"], fg=T["text"],
            insertbackground=T["accent"],
            selectbackground=T["accent_dim"],
            font=config.FONT_TEXT,
            wrap="word", bd=0, relief="flat",
            padx=10, pady=10,
        )
        self._txt_out.grid(row=1, column=0, sticky="nsew")
        self._txt_out.configure(state="disabled")

        self._wc_var = tk.StringVar(value="Words: 0  |  Chars: 0")
        tk.Label(
            right, textvariable=self._wc_var,
            font=config.FONT_LABEL_SM,
            bg=T["bg"], fg=T["text_dim"],
        ).grid(row=2, column=0, sticky="e", pady=(2, 4))

        _lbl(right, "Detected Entities", font=config.FONT_HEADING).grid(
            row=3, column=0, sticky="w", pady=(4, 3),
        )

        self._txt_ent = scrolledtext.ScrolledText(
            right,
            bg=T["bg_card"], fg=T["text"],
            font=config.FONT_TEXT,
            wrap="word", bd=0, relief="flat",
            padx=8, pady=8, height=9,
        )
        self._txt_ent.grid(row=4, column=0, sticky="nsew")
        self._txt_ent.configure(state="disabled")
        self._setup_entity_tags()

    def _setup_entity_tags(self):
        tags = {
            "email":    ("#38bdf8", "bold"),
            "phone":    ("#4ade80", "bold"),
            "date":     ("#fb923c", "bold"),
            "url":      ("#c084fc", "bold"),
            "currency": ("#facc15", "bold"),
            "label":    (T["text_dim"],   "italic"),
            "none":     (T["text_muted"], "normal"),
        }
        for tag, (color, weight) in tags.items():
            self._txt_ent.tag_configure(
                tag,
                foreground=color,
                font=(config.FONT_MONO, 10, weight),
            )

    # ── STATUS BAR ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = _frm(self, bg=T["bg_secondary"])
        bar.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Ready — upload an image or PDF to begin.")
        tk.Label(
            bar, textvariable=self._status_var,
            font=config.FONT_STATUS,
            bg=T["bg_secondary"], fg=T["text_dim"], anchor="w",
        ).pack(side="left", padx=12, pady=5)

        self._dot = tk.Label(
            bar, text="●",
            font=(config.FONT_FAMILY, 10),
            bg=T["bg_secondary"], fg=T["success"],
        )
        self._dot.pack(side="right", padx=12)

    # ── BUTTON STATE ──────────────────────────────────────────────────────────

    def _refresh_buttons(self):
        has_img  = bool(self._image_path)
        has_proc = self._processed_image is not None
        has_txt  = bool(self._extracted_text.strip())
        busy     = self._busy
        is_pdf   = self._is_pdf

        def _set(b, enabled):
            state = "normal" if (enabled and not busy) else "disabled"
            b.configure(
                state=state,
                fg=T["text_muted"] if state == "disabled" else T["text"],
            )

        _set(self._btn_upload,  True)
        _set(self._btn_pdf,     True)
        _set(self._btn_process, has_img)
        _set(self._btn_extract, has_proc)
        _set(self._btn_save,    has_txt)
        _set(self._btn_clear,   True)

        # PDF navigation
        if is_pdf and len(self._pdf_pages) > 1:
            self._pdf_nav.grid()
            self._btn_prev_page.configure(
                state="normal" if self._pdf_page_idx > 0 else "disabled"
            )
            self._btn_next_page.configure(
                state="normal" if self._pdf_page_idx < len(self._pdf_pages) - 1 else "disabled"
            )
            self._page_var.set(f"Page {self._pdf_page_idx + 1} of {len(self._pdf_pages)}")
        else:
            self._pdf_nav.grid_remove()

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            self._progress.start(14)
            self._dot.configure(fg=T["warning"])
        else:
            self._progress.stop()
            self._dot.configure(fg=T["success"])
        self._refresh_buttons()

    # ── STATUS HELPERS ────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    def _update_doc_badge(self, doc_type: str, confidence: float):
        colour_map = {
            "printed":     T["success"],
            "handwritten": T["accent"],
            "mixed":       T["warning"],
        }
        colour    = colour_map.get(doc_type.lower(), T["bg_card"])
        label     = doc_type.capitalize() if doc_type else "—"
        conf_text = f"({confidence * 100:.0f}% conf)" if confidence else ""
        self.after(0, lambda: self._doc_badge.set(label, colour))
        self.after(0, lambda: self._conf_lbl.configure(text=conf_text))

    def _update_accuracy_badge(self, pct: float):
        """Update the accuracy badge with colour coding."""
        if pct >= 90:
            color = T["success"]
        elif pct >= 75:
            color = T["warning"]
        else:
            color = T["error"]
        self.after(0, lambda: self._acc_badge.set(f"{pct:.1f}%", color))

    # ── CALLBACKS ─────────────────────────────────────────────────────────────

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self._load_image_file(path)

    def _on_upload_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self._load_pdf_file(path)

    def _load_image_file(self, path: str):
        try:
            img = load_image(path)
            self._reset_state()
            self._image_path     = path
            self._is_pdf         = False
            self._original_image = img
            self._panel_orig.set_image(img)
            self._status(f"Loaded: {os.path.basename(path)}  ({img.size[0]}×{img.size[1]} px)")
            self._refresh_buttons()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _load_pdf_file(self, path: str):
        self._set_busy(True)
        self._status(f"Loading PDF: {os.path.basename(path)}…")
        threading.Thread(target=self._worker_load_pdf, args=(path,), daemon=True).start()

    def _worker_load_pdf(self, path: str):
        try:
            from preprocessing.pdf_loader import PDFLoader
            pages = PDFLoader().load(path)

            if not pages:
                raise ValueError("No pages could be loaded from the PDF.")

            self._reset_state()
            self._image_path    = path
            self._is_pdf        = True
            self._pdf_pages     = pages
            self._pdf_page_idx  = 0

            # Display first page
            page_idx, page_img = pages[0]
            self._original_image = page_img
            self.after(0, lambda: self._panel_orig.set_image(page_img))
            self.after(0, lambda: self._status(
                f"PDF loaded: {os.path.basename(path)} — {len(pages)} page(s)"
            ))
        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"PDF load error: {err_msg}")
            self.after(0, lambda m=err_msg: messagebox.showerror("PDF Error", m))
            self.after(0, lambda m=err_msg: self._status(f"PDF load failed: {m}"))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(0, self._refresh_buttons)

    def _on_prev_page(self):
        if self._pdf_page_idx > 0:
            self._pdf_page_idx -= 1
            self._show_pdf_page()

    def _on_next_page(self):
        if self._pdf_page_idx < len(self._pdf_pages) - 1:
            self._pdf_page_idx += 1
            self._show_pdf_page()

    def _show_pdf_page(self):
        _, page_img = self._pdf_pages[self._pdf_page_idx]
        self._original_image  = page_img
        self._processed_image = None
        self._panel_orig.set_image(page_img)
        self._panel_proc.clear()
        self._refresh_buttons()
        self._status(f"Page {self._pdf_page_idx + 1} of {len(self._pdf_pages)}")

    def _on_process(self):
        if not self._image_path or self._busy:
            return
        self._set_busy(True)
        self._status("Preprocessing image…")
        threading.Thread(target=self._worker_process, daemon=True).start()

    def _worker_process(self):
        try:
            from preprocessing.image_cleaner import ImageCleaner
            from ocr.document_classifier     import DocumentClassifier
            from utils                        import cv2_to_pil

            cv_img         = ImageCleaner().preprocess(self._original_image)
            doc_type, conf = DocumentClassifier().classify(cv_img)
            proc_pil       = cv2_to_pil(cv_img)

            self._processed_image = proc_pil
            self._doc_type        = doc_type
            self._doc_confidence  = conf

            self.after(0, lambda: self._panel_proc.set_image(proc_pil))
            self.after(0, lambda: self._update_doc_badge(doc_type, conf))
            self.after(0, lambda: self._status(
                f"Processed — {doc_type.capitalize()} ({conf * 100:.0f}% confidence)"
            ))
        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"Process error: {err_msg}")
            self.after(0, lambda m=err_msg: messagebox.showerror("Processing Error", m))
            self.after(0, lambda m=err_msg: self._status(f"Error: {m}"))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(0, self._refresh_buttons)

    def _on_extract(self):
        if self._processed_image is None or self._busy:
            return
        self._set_busy(True)
        self._status("Running OCR — please wait…")
        threading.Thread(target=self._worker_extract, daemon=True).start()

    def _worker_extract(self):
        try:
            from ocr.hybrid_ocr_engine          import HybridOCREngine
            from postprocessing.enhanced_corrector import EnhancedCorrector
            from postprocessing.entity_extractor   import EntityExtractor

            # Run OCR
            result = HybridOCREngine().extract(
             self._processed_image, doc_type=self._doc_type
            )

            raw_text   = result["text"]
            confidence = result["confidence_pct"]

            corrector = EnhancedCorrector()
            clean     = corrector.process(
                raw_text,
                confidence=self._doc_confidence,
                doc_hint="auto",
            )

            # Accuracy report
            report = corrector.quality_report(raw_text, clean)
            acc    = report["confidence_pct"]
            self._accuracy = acc

            # Entity extraction
            ents = EntityExtractor().extract(clean)

            self._extracted_text = clean
            words = len(clean.split())
            chars = len(clean)

            self.after(0, lambda: self._display_text(clean))
            self.after(0, lambda: self._display_entities(ents))
            self.after(0, lambda: self._wc_var.set(f"Words: {words}  |  Chars: {chars}"))
            self.after(0, lambda: self._update_accuracy_badge(acc))
            self.after(0, lambda: self._status(
                f"Done — {words} words extracted. Estimated accuracy: {acc:.1f}%"
            ))

        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"Extract error: {err_msg}")
            self.after(0, lambda m=err_msg: messagebox.showerror("OCR Error", m))
            self.after(0, lambda m=err_msg: self._status(f"OCR Error: {m}"))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(0, self._refresh_buttons)

    def _on_save(self):
        if not self._extracted_text.strip():
            messagebox.showwarning("Nothing to save", "No text to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Text",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not path:
            return
        if save_text(self._extracted_text, path):
            self._status(f"Saved: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Text saved to:\n{path}")
        else:
            messagebox.showerror("Save Error", "Could not write file.")

    def _on_copy_text(self):
        if self._extracted_text.strip():
            self.clipboard_clear()
            self.clipboard_append(self._extracted_text)
            self._status("Text copied to clipboard.")

    def _on_clear(self):
        self._reset_state()
        self._panel_orig.clear()
        self._panel_proc.clear()
        self._clear_text_panels()
        self._update_doc_badge("", 0)
        self._wc_var.set("Words: 0  |  Chars: 0")
        self.after(0, lambda: self._acc_badge.set("—", T["bg_card"]))
        self._status("Cleared — ready for a new image or PDF.")
        self._refresh_buttons()

    # ── STATE RESET ───────────────────────────────────────────────────────────

    def _reset_state(self):
        self._image_path      = ""
        self._is_pdf          = False
        self._pdf_pages       = []
        self._pdf_page_idx    = 0
        self._original_image  = None
        self._processed_image = None
        self._extracted_text  = ""
        self._doc_type        = ""
        self._doc_confidence  = 0.0
        self._accuracy        = 0.0

    # ── TEXT DISPLAY ──────────────────────────────────────────────────────────

    def _display_text(self, text: str):
        self._txt_out.configure(state="normal")
        self._txt_out.delete("1.0", "end")
        self._txt_out.insert("end", text)
        self._txt_out.configure(state="disabled")

    def _display_entities(self, entities: dict):
        self._txt_ent.configure(state="normal")
        self._txt_ent.delete("1.0", "end")

        if not any(entities.values()):
            self._txt_ent.insert("end", "No entities detected.", "none")
        else:
            for etype, values in entities.items():
                if not values:
                    continue
                self._txt_ent.insert("end", f"{etype.upper()}:\n", "label")
                for v in values:
                    self._txt_ent.insert("end", f"  • {v}\n", etype)
                self._txt_ent.insert("end", "\n")

        self._txt_ent.configure(state="disabled")

    def _clear_text_panels(self):
        for w in (self._txt_out, self._txt_ent):
            w.configure(state="normal")
            w.delete("1.0", "end")
            w.configure(state="disabled")

    # ── RUN ───────────────────────────────────────────────────────────────────

    def run(self):
        self.mainloop()