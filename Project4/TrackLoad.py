def readTrackFile(trackType):

    fileName = trackType + "-track.txt"

    storedTrack = []

    with open(fileName, 'r') as file:
        # Read and discard the first line
        file.readline()

        for line in file:
            # Strip the newline character at the end of each line
            stripped_line = line.strip()
            # Split the line into individual characters and append to the 2D array
            storedTrack.append(list(stripped_line))

    return storedTrack
