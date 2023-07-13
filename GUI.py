import PySimpleGUI as sg
import os, logging, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
sg.change_look_and_feel('DefaultNoMoreNagging')  # for default look and feel


# Designing layout
layout = [[sg.Text("")], [sg.Text("\t\t\tEnter the Analysis\t"), sg.Combo(["Training data(%)", "K-Fold"], size=(20, 2))
                          , sg.InputText(size=(20, 2))],
           [sg.Text("\t"), sg.Button("START", size=(10, 2))],
           [sg.Button('Close', size=(10, 1))], [sg.Text("")]]

# Create the Window layout
window = sg.Window('GUI', layout)

# event loop
while True:
    event, value = window.read( )  # displays the window
    if event == "START":
        tr_per, cv = 0, 0
        Anal = value[0]
        if (value[0] == 'Training data(%)'):
            tr_per = int(value[1])
        else:
            cv = int(value[1])
        import Run
        Run.callmain(Anal, tr_per, cv)
        print("\nDone.!")

    if event == 'Close':
        window.close( )
        break