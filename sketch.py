from tkinter import *

canvas_width = 500
canvas_height = 500

def paint( event ):
   dot_size = 10
   x1, y1 = ( event.x - dot_size ), ( event.y - dot_size )
   x2, y2 = ( event.x + dot_size ), ( event.y + dot_size )
   w.create_oval( x1, y1, x2, y2, fill = '#000000' )
   message.configure(text="Elephant (90%)")


master = Tk()
master.title( "CPSC599 Quickdraw" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height,
           background='#FFFFFF')
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )
    
mainloop()