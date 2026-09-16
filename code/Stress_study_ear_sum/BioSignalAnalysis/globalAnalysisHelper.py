class GlobalHelper:
    @staticmethod
    def createSlidingWindows(data, sf, lookbackSec, stepSec, minStartSec):
        lookbackSamples = int(lookbackSec * sf)
        stepSamples = int(stepSec * sf)
        minStartSamples = int(minStartSec * sf)

        totalSamples = len(data)
        windows = []
        timestamps = []

        pointer = minStartSamples  # start pointer
        while pointer <= totalSamples:
            windowStart = pointer - lookbackSamples
            windowEnd = pointer
            if windowStart < 0:
                pointer += stepSamples
                continue
            if data.ndim == 1:
                windowData = data[windowStart:windowEnd]
            else:
                windowData = data[windowStart:windowEnd, :]
            windows.append(windowData)
            timestamps.append(pointer / sf)
            pointer += stepSamples

        return windows, timestamps
