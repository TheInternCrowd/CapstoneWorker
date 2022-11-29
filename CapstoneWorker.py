"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
import numpy as np
import mne
import tensorflow
import sys, os
import time
import keyboard
from game import dodgecar

#Load Model
buffer = np.zeros(shape = (100, 32))
model = tensorflow.keras.models.load_model("./model/frequencyFilteredRaw(Severalpeople)")

#load Game
dodgecar.game = dodgecar.Game()
dodgecar.game.__int__()

def main():
    #Connect to the EEG
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    while True:
        #Get Latest Reading from LSL (EEG Output)
        sample = inlet.pull_sample()
        latestReading = np.array(sample[0])

        # TEST ON RANDOM NUMBERS
        # latestReading = np.random.rand(32)

        #Shift & Update the Buffer
        for i in reversed(range(buffer.shape[0])):
            if(i != 0):
                buffer[i] = buffer[i - 1]
        buffer[0] = latestReading

        #Filter or Preprocess Data (This Depends on what the model is trained by)
        blockPrint()
        nedf = mne.io.read_raw_nedf(r".\config\20220927155814_Thomas_Alternating Fists.nedf", preload=True);
        nedfInfo = nedf.info
        tBuffer = np.zeros(shape = (33,100))
        tBuffer[0:32] = buffer.T
        rawArray = mne.io.RawArray(tBuffer, nedfInfo)
        modelInput = rawArray.filter(8, 45)[0:32][0].T #Ignore STI
        enablePrint()

        #Use Model Prediction From Current Buffer
        result = predict(buffer)[0]
        l = result[0]
        r = result[1]
        print("\n\nLEFT: ", l);
        print("RIGHT: ", r);

        #Update Game State Based on Prediction Result
        key = ""
        if(l < 0.5 and r < 0.5):
            key = "CENTER"
        elif(l > .5 and r < .5):
            key = "LEFT"
        elif(r > .5 and l < .5):
            key = "RIGHT"

        dodgecar.game.run_game_frame(key)

        print("============================================================")
        # print(buffer)
        time.sleep(.1);

def predict(currentBuffer):
    # return np.random.randint(2) #lol
    return model.predict(np.array([currentBuffer]))

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main()