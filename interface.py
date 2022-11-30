import project_utils

import tkinter as tk
import numpy as np
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to dir that contains the model.pb, keras_metadata.pb, and the subfolders /assets and /variables", required=True)
parser.add_argument("--memmory", help="This program iterates with two image files to update the changes on the Tk_window", required=True)
args = parser.parse_args()


# Tkinter window theme
themes = ['scidpink', 'scidsand', 'classic', 'keramik', 'itft1', 'scidpurple', 'yaru', 
          'clearlooks', 'black', 'blue', 'scidmint', 'breeze', 'plastik', 'ubuntu', 
          'adapta', 'kroc', 'arc', 'equilux', 'scidgreen', 'scidgrey', 'scidblue', 
          'clam', 'radiance', 'alt', 'default', 'smog', 'aquativo', 'winxpblue', 'elegance']

themes_to_use = ['scidblue', 'black', 'breeze', 'elegance']


# Loading the GAN model, setting the main path
images_path = args.memory
model = project_utils.model_load(args.model)


# root window
root = ThemedTk(theme=themes_to_use[1])
root.geometry('1100x320')
root.resizable(False, False)
root.title('Slider Demo')


latent_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype="float32")
latent_space = latent_space.reshape(1, 20)

# -------------------------------------------------------------------------------------
# ------------------------------- SLIDER CURRENT VALUES ------------------------------- 
# -------------------------------------------------------------------------------------

current_value1 = tk.DoubleVar()
current_value2 = tk.DoubleVar()
current_value3 = tk.DoubleVar()
current_value4 = tk.DoubleVar()
current_value5 = tk.DoubleVar()
current_value6 = tk.DoubleVar()
current_value7 = tk.DoubleVar()
current_value8 = tk.DoubleVar()
current_value9 = tk.DoubleVar()
current_value9 = tk.DoubleVar()
current_value10 = tk.DoubleVar()
current_value11 = tk.DoubleVar()
current_value12 = tk.DoubleVar()
current_value13 = tk.DoubleVar()
current_value14 = tk.DoubleVar()
current_value15 = tk.DoubleVar()
current_value16 = tk.DoubleVar()
current_value17 = tk.DoubleVar()
current_value18 = tk.DoubleVar()
current_value19 = tk.DoubleVar()
current_value20 = tk.DoubleVar()


# -------------------------------------------------------------------------------------
# --------------------------------- GET CURRENT VALUES -------------------------------- 
# -------------------------------------------------------------------------------------

def get_current_value1(current_value1):
    latent_space[0][0] = float( '{: .2f}'.format(current_value1.get()) )
    return '{: .2f}'.format(current_value1.get())

def get_current_value2(current_value2):
    latent_space[0][1] = float( '{: .2f}'.format(current_value2.get()) )
    return '{: .2f}'.format(current_value2.get())

def get_current_value3(current_value3):
    latent_space[0][2] = float( '{: .2f}'.format(current_value3.get()) )
    return '{: .2f}'.format(current_value3.get())

def get_current_value4(current_value4):
    latent_space[0][3] = float( '{: .2f}'.format(current_value4.get()) )
    return '{: .2f}'.format(current_value4.get())

def get_current_value5(current_value5):
    latent_space[0][4] = float( '{: .2f}'.format(current_value5.get()) )
    return '{: .2f}'.format(current_value5.get())

def get_current_value6(current_value6):
    latent_space[0][5] = float( '{: .2f}'.format(current_value6.get()) )
    return '{: .2f}'.format(current_value6.get())

def get_current_value7(current_value7):
    latent_space[0][6] = float( '{: .2f}'.format(current_value7.get()) )
    return '{: .2f}'.format(current_value7.get())

def get_current_value8(current_value8):
    latent_space[0][7] = float( '{: .2f}'.format(current_value8.get()) )
    return '{: .2f}'.format(current_value8.get())

def get_current_value9(current_value9):
    latent_space[0][8] = float( '{: .2f}'.format(current_value9.get()) )
    return '{: .2f}'.format(current_value9.get())

def get_current_value10(current_value10):
    latent_space[0][9] = float( '{: .2f}'.format(current_value10.get()) )
    return '{: .2f}'.format(current_value10.get())

def get_current_value11(current_value11):
    latent_space[0][10] = float( '{: .2f}'.format(current_value11.get()) )
    return '{: .2f}'.format(current_value11.get())

def get_current_value12(current_value12):
    latent_space[0][11] = float( '{: .2f}'.format(current_value12.get()) )
    return '{: .2f}'.format(current_value12.get())

def get_current_value13(current_value13):
    latent_space[0][12] = float( '{: .2f}'.format(current_value13.get()) )
    return '{: .2f}'.format(current_value13.get())

def get_current_value14(current_value14):
    latent_space[0][13] = float( '{: .2f}'.format(current_value14.get()) )
    return '{: .2f}'.format(current_value14.get())

def get_current_value15(current_value15):
    latent_space[0][14] = float( '{: .2f}'.format(current_value15.get()) )
    return '{: .2f}'.format(current_value15.get())

def get_current_value16(current_value16):
    latent_space[0][15] = float( '{: .2f}'.format(current_value16.get()) )
    return '{: .2f}'.format(current_value16.get())

def get_current_value17(current_value17):
    latent_space[0][16] = float( '{: .2f}'.format(current_value17.get()) )
    return '{: .2f}'.format(current_value17.get())

def get_current_value18(current_value18):
    latent_space[0][17] = float( '{: .2f}'.format(current_value18.get()) )
    return '{: .2f}'.format(current_value18.get())

def get_current_value19(current_value19):
    latent_space[0][18] = float( '{: .2f}'.format(current_value19.get()) )
    return '{: .2f}'.format(current_value19.get())

def get_current_value20(current_value20):
    latent_space[0][19] = float( '{: .2f}'.format(current_value20.get()) )
    return '{: .2f}'.format(current_value20.get())



# -------------------------------------------------------------------------------------
# ----------------------------------- SLIDER CHANGES ---------------------------------- 
# -------------------------------------------------------------------------------------

def slider_changed1(event):
    value_label1.configure(text=get_current_value1(current_value1))

def slider_changed2(event):
    value_label2.configure(text=get_current_value2(current_value2))

def slider_changed3(event):
    value_label3.configure(text=get_current_value3(current_value3))

def slider_changed4(event):
    value_label4.configure(text=get_current_value4(current_value4))

def slider_changed5(event):
    value_label5.configure(text=get_current_value5(current_value5))

def slider_changed6(event):
    value_label6.configure(text=get_current_value6(current_value6))

def slider_changed7(event):
    value_label7.configure(text=get_current_value7(current_value7))

def slider_changed8(event):
    value_label8.configure(text=get_current_value8(current_value8))

def slider_changed9(event):
    value_label9.configure(text=get_current_value9(current_value9))

def slider_changed10(event):
    value_label10.configure(text=get_current_value10(current_value10))

def slider_changed11(event):
    value_label11.configure(text=get_current_value11(current_value11))

def slider_changed12(event):
    value_label12.configure(text=get_current_value12(current_value12))

def slider_changed13(event):
    value_label13.configure(text=get_current_value13(current_value13))

def slider_changed14(event):
    value_label14.configure(text=get_current_value14(current_value14))

def slider_changed15(event):
    value_label15.configure(text=get_current_value15(current_value15))

def slider_changed16(event):
    value_label16.configure(text=get_current_value16(current_value16))

def slider_changed17(event):
    value_label17.configure(text=get_current_value17(current_value17))

def slider_changed18(event):
    value_label18.configure(text=get_current_value18(current_value18))

def slider_changed19(event):
    value_label19.configure(text=get_current_value19(current_value19))

def slider_changed20(event):
    value_label20.configure(text=get_current_value20(current_value20))

    
# -------------------------------------------------------------------------------------
# -------------------- SLIDERS TO CHANGE THE VALUES ON LATENT SPACE ------------------- 
# -------------------------------------------------------------------------------------

slide_width = 120
slide_height = 20

# slider 1
slider1 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed1,    
    variable=current_value1
)
slider1.place( x=50, y=50, width=slide_width, height=slide_height )

# slider 2
slider2 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed2,    
    variable=current_value2
)
slider2.place( x=50, y=100, width=slide_width, height=slide_height )

# slider 3
slider3 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed3,    
    variable=current_value3
)
slider3.place( x=50, y=150, width=slide_width, height=slide_height )

# slider 4
slider4 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed4,    
    variable=current_value4
)
slider4.place( x=50, y=200, width=slide_width, height=slide_height )

# slider 5
slider5 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed5,    
    variable=current_value5
)
slider5.place( x=50, y=250, width=slide_width, height=slide_height )

# slider 6
slider6 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed6,    
    variable=current_value6
)
slider6.place( x=200, y=50, width=slide_width, height=slide_height )

# slider 7
slider7 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed7,    
    variable=current_value7
)
slider7.place( x=200, y=100, width=slide_width, height=slide_height )

# slider 8
slider8 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed8,    
    variable=current_value8
)
slider8.place( x=200, y=150, width=slide_width, height=slide_height )

# slider 9
slider9 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed9,    
    variable=current_value9
)
slider9.place( x=200, y=200, width=slide_width, height=slide_height )

# slider 10
slider10 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed10,    
    variable=current_value10
)
slider10.place( x=200, y=250, width=slide_width, height=slide_height )


# slider 11
slider11 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed11,    
    variable=current_value11
)
slider11.place( x=350, y=50, width=slide_width, height=slide_height )

# slider 12
slider12 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed12,    
    variable=current_value12
)
slider12.place( x=350, y=100, width=slide_width, height=slide_height )

# slider 13
slider13 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed13,    
    variable=current_value13
)
slider13.place( x=350, y=150, width=slide_width, height=slide_height )

# slider 14
slider14 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed14,    
    variable=current_value14
)
slider14.place( x=350, y=200, width=slide_width, height=slide_height )

# slider 15
slider15 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed15,    
    variable=current_value15
)
slider15.place( x=350, y=250, width=slide_width, height=slide_height )

# slider 16
slider16 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed16,    
    variable=current_value16
)
slider16.place( x=500, y=50, width=slide_width, height=slide_height )

# slider 17
slider17 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed17,    
    variable=current_value17
)
slider17.place( x=500, y=100, width=slide_width, height=slide_height )

# slider 18
slider18 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed18,    
    variable=current_value18
)
slider18.place( x=500, y=150, width=slide_width, height=slide_height )

# slider 19
slider19 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed19,    
    variable=current_value19
)
slider19.place( x=500, y=200, width=slide_width, height=slide_height )

# slider 20
slider20 = ttk.Scale(
    root,
    from_=-1.,
    to=1.,
    orient='horizontal',  # vertical
    command=slider_changed20,    
    variable=current_value20
)
slider20.place( x=500, y=250, width=slide_width, height=slide_height )


# -------------------------------------------------------------------------------------
#  ---------------------- SHOW THE CURRENT VALUE FOR EACH SLIDER ---------------------- 
# -------------------------------------------------------------------------------------

value_label_width = 40
value_label_height = 20

# value label1
value_label1 = ttk.Label(
    root,
    text=get_current_value1(current_value1)
)
value_label1.place( x=90, y = 30, width = value_label_width, height= value_label_height )

# value label2
value_label2 = ttk.Label(
    root,
    text=get_current_value2(current_value2)
)
value_label2.place( x=90, y = 80, width = value_label_width, height= value_label_height )

# value label3
value_label3 = ttk.Label(
    root,
    text=get_current_value3(current_value3)
)
value_label3.place( x=90, y = 130, width = value_label_width, height= value_label_height )

# value label4
value_label4 = ttk.Label(
    root,
    text=get_current_value4(current_value4)
)
value_label4.place( x=90, y = 180, width = value_label_width, height= value_label_height )

# value label5
value_label5 = ttk.Label(
    root,
    text=get_current_value5(current_value5)
)
value_label5.place( x=90, y = 230, width = value_label_width, height= value_label_height )

# value label6
value_label6 = ttk.Label(
    root,
    text=get_current_value6(current_value6)
)
value_label6.place( x=240, y = 30, width = value_label_width, height= value_label_height )

# value label7
value_label7 = ttk.Label(
    root,
    text=get_current_value7(current_value7)
)
value_label7.place( x=240, y = 80, width = value_label_width, height= value_label_height )

# value label8
value_label8 = ttk.Label(
    root,
    text=get_current_value8(current_value8)
)
value_label8.place( x=240, y = 130, width = value_label_width, height= value_label_height )

# value label9
value_label9 = ttk.Label(
    root,
    text=get_current_value9(current_value9)
)
value_label9.place( x=240, y = 180, width = value_label_width, height= value_label_height )

# value label10
value_label10 = ttk.Label(
    root,
    text=get_current_value10(current_value10)
)
value_label10.place( x=240, y = 230, width = value_label_width, height= value_label_height )

# value label11
value_label11 = ttk.Label(
    root,
    text=get_current_value11(current_value11)
)
value_label11.place( x=390, y = 30, width = value_label_width, height= value_label_height )

# value label12
value_label12 = ttk.Label(
    root,
    text=get_current_value12(current_value12)
)
value_label12.place( x=390, y = 80, width = value_label_width, height= value_label_height )

# value label13
value_label13 = ttk.Label(
    root,
    text=get_current_value13(current_value13)
)
value_label13.place( x=390, y = 130, width = value_label_width, height= value_label_height )

# value label14
value_label14 = ttk.Label(
    root,
    text=get_current_value14(current_value14)
)
value_label14.place( x=390, y = 180, width = value_label_width, height= value_label_height )

# value label15
value_label15 = ttk.Label(
    root,
    text=get_current_value15(current_value15)
)
value_label15.place( x=390, y = 230, width = value_label_width, height= value_label_height )

# value label16
value_label16 = ttk.Label(
    root,
    text=get_current_value16(current_value16)
)
value_label16.place( x=540, y = 30, width = value_label_width, height= value_label_height )

# value label17
value_label17 = ttk.Label(
    root,
    text=get_current_value17(current_value17)
)
value_label17.place( x=540, y = 80, width = value_label_width, height= value_label_height )

# value label18
value_label18 = ttk.Label(
    root,
    text=get_current_value18(current_value18)
)
value_label18.place( x=540, y = 130, width = value_label_width, height= value_label_height )

# value label19
value_label19 = ttk.Label(
    root,
    text=get_current_value19(current_value19)
)
value_label19.place( x=540, y = 180, width = value_label_width, height= value_label_height )

# value label20
value_label20 = ttk.Label(
    root,
    text=get_current_value20(current_value20)
)
value_label20.place( x=540, y = 230, width = value_label_width, height= value_label_height )

#current_value_label = ttk.Label(
#    root,
#    text='Current Value:'
#)
#
#current_value_label.place(
#    x=190,
#    y = 80,
#    width = 40,
#    height= 20
#)

# -------------------------------------------------------------------------------------
# ------------------------------------ IMAGE LABELS ----------------------------------- 
# -------------------------------------------------------------------------------------

### Default image

default_path = images_path + "default_img.jpg"

project_utils.latentSpace2imgJpg(model=model, latent_space=latent_space, image_path=default_path)
def_img = Image.open(default_path)
default = ImageTk.PhotoImage(def_img)

default_generated_label = ttk.Label(root, image=default)
default_generated_label.place( x=880, y = 60)

last_gen = ImageTk.PhotoImage(def_img)

# Image that will change whe we set new values to the latent space
gen_img_label = ttk.Label(root, image=default)
gen_img_label.place( x=680, y = 60)

# Function to change image on that label "gen_img_label"
def imageelection():
    global latent_space, default, newGen, last_gen
    newGen_path = images_path + "newGen_img.jpg"

    project_utils.latentSpace2imgJpg(model=model, latent_space=latent_space, image_path=newGen_path)
    newGen = Image.open(newGen_path)
    newGen = ImageTk.PhotoImage(newGen)
    
    gen_img_label.configure(image=newGen)
    default_generated_label.configure(image=default)

    image_sup = newGen
    newGen = last_gen
    last_gen = image_sup

# Button to change the image
gen_button=ttk.Button(root,text="Generate",command=imageelection)
gen_button.place( x=680, y=240 )

root.mainloop()