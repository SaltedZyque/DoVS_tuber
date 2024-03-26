# Project Concept and Objectives

> To see full process of implementation, see [Imp_Doc](Imp_Doc.md)

The purpose of this project is to create a personal live facial tracking software than can be linked to an animated model, allowing a user to control a virtual character's face using their own - including features like head movements, eyes and speech. 

This project essentially emulates the process of a [VTuber](https://en.wikipedia.org/wiki/VTuber) (Virtual Youtuber, known for creating internet content through an animated avatar) setup. This comes in two parts: animation / design of the model, and capturing user facial features to manipulate the model itself.

While the modelling process for VTubers are well documented with a lot of ways to customize, the facial tracking software used are generally behind a paywall or not open source. The goal of this project is to create a facial tracking software that is open source, customizable and compatible with existing modelling software.

## Checklist
A checklist of goals to be accomplished for this project is shown as follows. They include incomplete stretch goals that were considered in case I had more time with the project.

- [x] Static Image Facial Detection 
- [x] Live Face Detection through Webcam  
- [x] Precise Facial Feature Tracking - Landmark Detection with Dlib
- [ ] Identify and Parameterize Specific Facial Features
	- [x] Eyes - Open / Close
	- [ ] Head Movement - Left / Right (X), Up / Down (Y), Rotation
	- [ ] Eyebrows - Raise / Normal
	- [ ] Mouth (Jaw) - Open / Close
	- [ ] Lips - Smile / Frown
- [ ] Connect Tracking to a Model (2D) -- Stretch Goal
	- [ ] **Design a simple model** that can move individual parts (or use a template model available for free)
	- [ ] Create a **Unity build that communicates** with the Python files for Parameter Control - likely OSC socket connection
	- [ ] **Normalize and parse input values** from Tracking Data to Model Parameters (i.e. JSON file of all parameter values to send)
	- [ ] Unity build should link to model based on Live2D Unity SDK - allowing for initial calibration, movement and facial features
- [ ] Implement to a streaming software to capture model (OBS)

# Run Instructions
1. Ensure Webcam is On
2. Install required modules (with pip install)
3. Run [face_tracker.py](face_tracker.py)
## Modules Used
- OpenCV
- Matplotlib (optional)
- Dlib (CMake and [Visual C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/) install needed)
- Numpy
- Scipy
- Imutils
# Project Demo

Running [dlib_landmarks.py](py/dlib_landmarks.py)
![](assets/doc/live2.mp4)

Running [face_tracker.py](face_tracker.py)
![](assets/doc/eyetracking.mp4)
# Evaluation and Next Steps

The final Python file is able to:
- Detect the presence of a face in live video
- Map out specific landmark features of the face (including eyes, nose, jaw, lips)
- Change a parameter when it detects that the face has closed their eye(s)

Things it is unable to do but should be worked on for future iterations:
- Complete the parameter detection for head movements, mouth movements and eyebrows
- Be integrated into a Unity environment through an OSC connection
- Control a virtual model in Unity using the Live2D Unity SDK

Further Steps for the project may include:
- Optimising the tracking using neural networks, calibration and smoothing
- Adding extra customization features like emotion recognition algorithms or multiple face tracking

I have also researched into projects that have been similar to mine, which show what the completed version of this project would look like:
Example Projects with Full Implementation
https://github.com/devmiren/MirenStudio
https://github.com/factaxd/Unity-Live2D-Facetracking/tree/2288373311a1d52b096d9f176258ec7ac5b991a7

# Reflection
This was a very fun project! Unfortunately I had to finish it individually as my teammate will be taking an interruption of studies, and as such I was not able to complete as much of the project as I would have liked to. 

Despite this, it has been interesting transitioning to using Python instead of Matlab for this project, as Python has so many readily available modules that allowed the coding to be easier to use. I followed a couple of different tutorials to learn how the facial tracking works, and was able to figure out my own way to indicate the blinking of the eyes. I might potentially explore using my own trained facial detection neural network in the future, just to learn how to implement it.

I hoped I had more time to develop the Unity side of this project, however as I did not have the skillset readily available and not enough time, this section of the project was not accomplished. 

In terms of mistakes made during the project - I think I focused too much on researching how the models are supposed to be made and animated, when I could have just used a template model for this project. I also did not realize making my software on Unity was the only way to accomplish what I wanted in terms of manipulating the model, and did not factor in the time I would have needed for it. 