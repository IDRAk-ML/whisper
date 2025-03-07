import asyncio
import websockets
import sys

async def send_audio_and_receive_transcript(uri, file_path):
    async with websockets.connect(uri) as websocket:
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
            
        await websocket.send(audio_data)
        transcript = await websocket.recv()
        print("Transcript:", transcript)

async def send_persistent_audio(uri, file_path):
    print('presistent')
    async with websockets.connect(uri) as websocket:
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
            
        await websocket.send(audio_data)
        while True:
            transcript = await websocket.recv()
            print("Transcript:", transcript)



if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python client.py <audio_file.wav>")
    #     sys.exit(1)
    
    file_path = '/Users/ali/Downloads/am_21.wav'
    websocket_uri = "ws://148.251.178.29:9005/ws_file_transcribe1"  # Adjust this if the server is running on a different host/port
    
    asyncio.run(send_audio_and_receive_transcript(websocket_uri, file_path))