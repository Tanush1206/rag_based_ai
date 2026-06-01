# convert videos to mp3
import os
import subprocess

files = os.listdir("videos")
for file in files :
    if file.startswith("."):
        continue
    
    title = file.replace(" | ", " _ ").split(" _ ")[0]
    tutorial_num = file.split(" #")[1].replace(".mp4", "")
    
    print(f"Title: {title}")
    print(f"Tutorial Number: {tutorial_num}")
    print("-" * 20)

    subprocess.run(["ffmpeg" , "-y", "-i" , f"videos/{file}" , f"audios/{tutorial_num}_{title}.mp3"])