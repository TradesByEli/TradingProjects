
import tkinter as tk
import pytz
from datetime import datetime
# Timezone
market_tz = pytz.timezone('US/Eastern')
def time():
    now = datetime.now(market_tz)
    current_time = now. strftime("%I:%M:%S %p")
    label.config(text=current_time)
    label.after(1000, time)
# Tkinter Window
root = tk.Tk()
root.title("Live Market Clock")
root.geometry("250x100")
root.configure(bg="black")

# Clock Label
label= tk.Label(root, font=('Digital-7', 48), fg="white", bg="black")
label.pack(anchor="center")

#  Start Clock
time()
root.mainloop()