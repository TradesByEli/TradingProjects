import tkinter as tk
from tkinter import font
from datetime import datetime

import pytz

# Timezone
market_tz = pytz.timezone("US/Eastern")

PREFERRED_FONTS = [
    "Digital-7",
    "Consolas",
    "Courier New",
    "Arial",
    "Helvetica",
    "Times New Roman",
    "Verdana",
    "Georgia",
    "Tahoma",
    "Segoe UI",
    "Calibri",
]

# Default display settings
clock_settings = {
    "font_family": "Arial",
    "font_size": 30,
    "label_width": 15,
    "text_color": "white",
    "transparent_mode": True,
    "stay_on_top": True,
}

update_job_id = None
settings_window = None


def get_available_fonts():
    """Return the list of installed font families."""
    try:
        if tk._default_root is None:
            return []
        return sorted(set(font.families(root=tk._default_root)))
    except tk.TclError:
        return []


def resolve_font_family(font_name: str) -> str:
    """Return a safe font family that is available on the current system."""
    available_fonts = get_available_fonts()
    if font_name in available_fonts:
        return font_name

    for preferred in PREFERRED_FONTS:
        if preferred in available_fonts:
            return preferred

    return "TkDefaultFont"


def update_clock():
    """Refresh the displayed market time every second."""
    global update_job_id

    now = datetime.now(market_tz)
    current_time = now.strftime("%I:%M:%S %p")
    label.config(text=current_time)
    update_job_id = root.after(1000, update_clock)


def apply_settings(font_family: str, font_size: int, label_width: int, text_color: str):
    """Apply user-selected settings to the clock display."""
    safe_font = resolve_font_family(font_family)
    label.config(font=(safe_font, font_size), width=label_width, fg=text_color)
    clock_settings.update(
        {
            "font_family": safe_font,
            "font_size": font_size,
            "label_width": label_width,
            "text_color": text_color,
        }
    )


def apply_window_topmost(enabled: bool):
    """Toggle whether the clock window stays above other windows."""
    clock_settings["stay_on_top"] = enabled
    root.wm_attributes("-topmost", enabled)


def apply_transparent_mode(enabled: bool):
    """Toggle text-only transparent mode for the clock window."""
    clock_settings["transparent_mode"] = enabled

    if enabled:
        settings_button.pack_forget()
        label.configure(cursor="fleur", bg="black")
        root.overrideredirect(True)
        try:
            root.wm_attributes("-transparentcolor", "black")
        except tk.TclError:
            pass
    else:
        root.overrideredirect(False)
        label.configure(cursor="", bg="black")
        try:
            root.wm_attributes("-transparentcolor", "")
        except tk.TclError:
            pass
        settings_button.pack(anchor="center")


def open_settings():
    """Open a settings window to customize the clock look and width."""
    global settings_window

    if settings_window is not None and settings_window.winfo_exists():
        settings_window.lift()
        settings_window.focus_force()
        return

    settings_window = tk.Toplevel(root)
    settings_window.title("Clock Settings")
    settings_window.configure(bg="black")
    settings_window.resizable(False, False)

    def close_settings_window():
        global settings_window
        settings_window.destroy()
        settings_window = None

    settings_window.protocol("WM_DELETE_WINDOW", close_settings_window)

    tk.Label(settings_window, text="Font Family", fg="white", bg="black").grid(
        row=0, column=0, padx=10, pady=6, sticky="w"
    )

    available_fonts = get_available_fonts()
    font_choices = [family for family in PREFERRED_FONTS if family in available_fonts]

    current_font = clock_settings["font_family"]
    if current_font in available_fonts and current_font not in font_choices:
        font_choices.insert(0, current_font)
    if not font_choices:
        font_choices = ["TkDefaultFont"]

    selected_font = tk.StringVar(value=resolve_font_family(current_font))
    tk.OptionMenu(settings_window, selected_font, *font_choices).grid(
        row=0, column=1, padx=10, pady=6
    )

    tk.Label(settings_window, text="Text Size", fg="white", bg="black").grid(
        row=1, column=0, padx=10, pady=6, sticky="w"
    )
    font_size_var = tk.StringVar(value=str(clock_settings["font_size"]))
    tk.Entry(settings_window, textvariable=font_size_var, width=10).grid(
        row=1, column=1, padx=10, pady=6
    )

    tk.Label(settings_window, text="Text Box Width", fg="white", bg="black").grid(
        row=2, column=0, padx=10, pady=6, sticky="w"
    )
    label_width_var = tk.StringVar(value=str(clock_settings["label_width"]))
    tk.Entry(settings_window, textvariable=label_width_var, width=10).grid(
        row=2, column=1, padx=10, pady=6
    )

    tk.Label(settings_window, text="Text Color", fg="white", bg="black").grid(
        row=3, column=0, padx=10, pady=6, sticky="w"
    )
    text_color_var = tk.StringVar(value=clock_settings["text_color"])
    tk.Entry(settings_window, textvariable=text_color_var, width=10).grid(
        row=3, column=1, padx=10, pady=6
    )

    transparent_mode_var = tk.BooleanVar(value=clock_settings["transparent_mode"])
    tk.Checkbutton(
        settings_window,
        text="Text only (transparent window)",
        variable=transparent_mode_var,
        fg="white",
        bg="black",
        selectcolor="#1f1f1f",
        activebackground="black",
        activeforeground="white",
    ).grid(row=4, column=0, columnspan=2, padx=10, pady=6, sticky="w")

    stay_on_top_var = tk.BooleanVar(value=clock_settings["stay_on_top"])
    tk.Checkbutton(
        settings_window,
        text="Window stay on top",
        variable=stay_on_top_var,
        fg="white",
        bg="black",
        selectcolor="#1f1f1f",
        activebackground="black",
        activeforeground="white",
    ).grid(row=5, column=0, columnspan=2, padx=10, pady=6, sticky="w")

    message_var = tk.StringVar(value="")
    tk.Label(settings_window, textvariable=message_var, fg="#ff6961", bg="black").grid(
        row=7, column=0, columnspan=2, pady=(0, 8)
    )

    def save_settings():
        try:
            new_font_size = int(font_size_var.get())
            new_label_width = int(label_width_var.get())
            if new_font_size <= 0 or new_label_width <= 0:
                raise ValueError
        except ValueError:
            message_var.set("Text size and width must be positive integers.")
            return

        new_text_color = text_color_var.get().strip()
        if not new_text_color:
            message_var.set("Text color is required.")
            return

        try:
            label.configure(fg=new_text_color)
        except tk.TclError:
            message_var.set("Text color must be a valid color name or hex code.")
            return

        apply_settings(selected_font.get(), new_font_size, new_label_width, new_text_color)
        apply_transparent_mode(transparent_mode_var.get())
        apply_window_topmost(stay_on_top_var.get())
        close_settings_window()

    tk.Button(
        settings_window,
        text="Apply",
        command=save_settings,
        bg="#1f1f1f",
        fg="white",
        width=12,
    ).grid(row=6, column=0, columnspan=2, pady=8)


def start_window_drag(event):
    """Start dragging the undecorated window."""
    if not clock_settings["transparent_mode"]:
        return
    root._drag_start_x = event.x_root
    root._drag_start_y = event.y_root
    root._window_start_x = root.winfo_x()
    root._window_start_y = root.winfo_y()


def drag_window(event):
    """Move the undecorated window with the mouse."""
    if not clock_settings["transparent_mode"]:
        return
    delta_x = event.x_root - root._drag_start_x
    delta_y = event.y_root - root._drag_start_y
    root.geometry(f"+{root._window_start_x + delta_x}+{root._window_start_y + delta_y}")


# Tkinter Window
root = tk.Tk()
root.title("Live Market Clock")
root.geometry("320x130")
root.configure(bg="black")

# Clock Label
initial_font = resolve_font_family(clock_settings["font_family"])
clock_settings["font_family"] = initial_font
label = tk.Label(
    root,
    font=(initial_font, clock_settings["font_size"]),
    fg=clock_settings["text_color"],
    bg="black",
    width=clock_settings["label_width"],
)
label.pack(anchor="center", pady=(18, 8))
label.bind("<ButtonPress-1>", start_window_drag)
label.bind("<B1-Motion>", drag_window)
label.bind("<Double-Button-1>", lambda _event: open_settings())
root.bind("<ButtonPress-1>", start_window_drag, add="+")
root.bind("<B1-Motion>", drag_window, add="+")

settings_button = tk.Button(
    root,
    text="Settings",
    command=open_settings,
    bg="#1f1f1f",
    fg="white",
)
settings_button.pack(anchor="center")

# Apply default window behaviors
apply_window_topmost(clock_settings["stay_on_top"])
apply_transparent_mode(clock_settings["transparent_mode"])

# Start Clock
update_clock()
root.mainloop()

# .Exe File
# pyinstaller --onefile --noconsole --hidden-import=pytz "C:\Program Files\PycharmProjects\Trading Projects\LiveClock.py"
