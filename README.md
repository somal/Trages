Trages
======

We are command of students from N.Novgorod, Russia. We wrote application which translate gestures via simple camera to text or speech. And after that we wrote GUI using beatiful framework Kivy. So we are particapating in Kivy Contest #2.

Main aim:
Mainly the application aim to deaf and mute children (other deaf-mute people can use it too!).
They have great diffucalties in life and in education. We told with headmaster of N.Novgorod deaf-mute school who teach with children about 20 years using own innovative methods. And he told that this application is very important for his and children. So it's very important in EDUCATION.

- Attention:
For working you should download, unpack source/pyc files.rar and add this files into folder with application (GitHub requirement).

======
Requrements:
- simple camera ( It can be web camera, embedded camera in smartphone etc)
- constant camera position
- only computer's platform (Windows, Linux, Mac OS)
- OpenCV (library of computer vision; http://opencv.org/downloads.html )
- Kivy (library of GUI; http://kivy.org/#download)  
- pyttsx (library of text-to-speech; https://pypi.python.org/pypi/pyttsx)
- NumPy (library for work with arrays; http://www.numpy.org/)
- demo version of recognition (We add this application, source of code with using of KIVY,but we don't want to add All recognition. So we add only 4 letters for the beginning ('A','L','C','H')


Advantages:
- cross-platform (Windows, Linux, Mac OS)
- high-accuracy application (about 95%)
- working in real-time mode (big speed of application)
- not sensitive to different external conditions (mainly, background and illumination level)
- comfortable working
- affortabling
- colorful design (It's veru important for deaf and mute children)

======
Step by step:
- After run application we show loading with our background. Then we click on image.
- We see 4 options with description: Children Education, Game, About us, Exit.
- If we click to 'Children Education' then we can run it. 
(Children can be teached using this option. They see default letter and try to show gesture letter.)
From the beginning you initialize your hand position. You should move hand that IT INCLUDED HAND.
After that we see INTERNATIONAL GESTURE ALPHABET, default alphabet, image via camera and result image. 
Blue running rectangle must track hand position (if we have got bug with it then please, move hand in this rectangle).
If you show gesture like in alphabet then software will recognize it and show in images.
For end we should press 'Escape' and we will hear text-to-speech of all previous letters.
- If we click to 'Game' then we can run it 
(After education cildren can be tested in gaming format.)
It have got the same design and control,but...
We see on the image via camera and follow to instuctions. For example, we show defined gesture and software show how right we did it.
- Also we can read about us and exit.


=======
If you have any questions then you can write to email:somal1996@rambler.ru;

