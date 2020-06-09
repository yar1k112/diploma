import collections
import contextlib
import sys
import wave
import math
import webrtcvad
import noisereduce as nr
import scipy as sp
import numpy as np
import librosa
import time
import os
from pocketsphinx import get_data_path, get_model_path, Pocketsphinx


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        # assert num_channels == 1
        sample_width = wf.getsampwidth()
        # assert sample_width == 2
        sample_rate = wf.getframerate()
        # assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        if(num_channels == 2):
            pcm_data = pcm_data[1::2]
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    res = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                res.append(ring_buffer[0][0].timestamp)
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                res.append(frame.timestamp)
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                ring_buffer.clear()
    if triggered:
        res.append(frame.timestamp)
    return res

def getConfig():

    model_path = os.getcwd()
    print(model_path)

    return {
        "hmm":os.path.join(model_path, 'zero_ru.cd_cont_4000'),
        # "lm":os.path.join(model_path, 'voxforge_ru.lm.bin'),
        "lm":os.path.join(model_path, "ru.lm"),
        "dic":os.path.join(model_path, 'ru.dic')
    }

def process(filePath, ps, f, verbose):
    audio, sample_rate = read_wave(filePath)
    vad = webrtcvad.Vad(mode=3)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 15, 100, vad, frames)
    segments = [math.floor(i * 1000) for i in segments]
    data, rate = librosa.load(filePath)

    # select section of data that is noise
    noisy_part = np.array([])
    i = 0
    segments = [math.floor(i*(rate/1000)) for i in segments]
    print(segments)
    if(len(segments) % 2 == 0):
        segments.insert(0,0)
        while i < len(segments) - 1:
            print(segments[i],segments[i+1])
            tmp = data[segments[i]:segments[i+1]]
            noisy_part = np.concatenate((noisy_part, tmp),axis=None)
            i += 2
    else:
        noisy_part = [0, segments[0]]
    
    # perform noise reduction
    tmpTime = time.time()
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=verbose)
    tmpTime = time.time() - tmpTime

    sp.io.wavfile.write("out.wav",rate,reduced_noise)
    os.system('ffmpeg -i out.wav -ar 16000 out-.wav')

    f.write("File:" + filePath + "\n")

    print("Start to recognizing...\n")
    realTime = time.time()
    ps.decode(
        audio_file='out-.wav',
        buffer_size=2048,
        no_search=False,
        full_utt=False
    )
    realTime = tmpTime + (time.time() - realTime)

    f.write(ps.hypothesis() + "\n")

    print("Recognized\n\n")

    f.write("Time:" + str(realTime) + "\n\n")

    os.system("del out-.wav")


def main(args):
    # if not args:
    #     print("args are required")
    #     exit(0)

    config = getConfig()
    ps = Pocketsphinx(**config)

    # if (args[0] == '--test'):
    #     withGraphics = False
    #     testsRootDir = "./../tests"
    #     resultsDir = "./../testResults"

    #     for dirName in os.listdir(testsRootDir):

    #         for filename in os.listdir(testsRootDir + "/" + dirName):
    #             path = testsRootDir + "/" + dirName + "/" + filename

    #             if(dirName == "indfrdic"):
    #                 for deepFile in os.listdir(path):
    #                     print(f"I'm in if {path}/{deepFile} and file name: {filename}")
    #                     f = open(resultsDir + f"/{dirName}_{filename}_w_filter_results.txt",'a')
    #                     process(path + "/" + deepFile,ps,f, withGraphics)
    #             else:
    #                 print(f"I'm in else {path}")
    #                 f = open(resultsDir + f"/{dirName}_w_filter_results.txt",'a')
    #                 process(path,ps,f, withGraphics)
    # elif (args[0] == '-P'):
    withGraphics = True
    f = open('result.txt', 'a')
    process(args[0], ps, f, withGraphics)


        

if __name__ == '__main__':
    main(sys.argv[1:])