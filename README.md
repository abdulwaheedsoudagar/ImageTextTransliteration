# ImageTextTranslation
![io1](https://user-images.githubusercontent.com/20552376/102013812-74360800-3d78-11eb-9b00-e0921528701f.jpg)<br>
![io2](https://user-images.githubusercontent.com/20552376/102013849-b19a9580-3d78-11eb-8d69-c3a7d1fd2002.jpg)

**Task here is to transliteration of english words present in images.**<br><br>
It consists of three sub-task:-<br><br>
1. Detection of english word in image.<br>
2. Converting the detected word(crop images) in image to text.<br>
3. Transliteration of each each text word from above step.

*Step 1*
# Detection of english word in image.<br>
The first step is to identify the location of English text in the image, here I used detectron for text detection. It should not recognize any other language text like Hindi etc. I have trained a model with a dataset of two languages (English and Hindi) with nearly 200 images, Model accurately identifies between English and Hindi. Model tris to put identified text into these classes if the text of any other language is detected. <br>
The structure of the project is Lil messed up, initially, I was training using my laptop, but for object detection, I had to collab, since for windows detection is not supported. I didn't want to face any trouble later.:sweat_smile:

*Step 2*
# Converting the detected word(crop images) in image to text
This model's architecture is taken from https://arxiv.org/pdf/1507.05717.pdf. <br>
Once the english text is detected, the model will extract the features from the image, either pretrained model or custom feature architecture can be used. Details are present in the paper.

*Step 3*
# Transliteration of each each text word from above step.
Sequence to Sequence model with Attention Mechanism is used for Transliteration. Two layers of GRU is used in this model. Thanks to padhAI online course for prvoding the starter code.<br>

Once all the models are trained, it can be used for Transliteration.


**Limitations**
* Input image should have either english or hindi text, as the model is trained only on those images.
* Accurecy will improve by bring in more data.
* Strucure of this repo is little messed up, :sweat_smile: :sweat_smile:
* Converting curved text on images doesn't work that well, I had worked little to straightening the text using openCV, still working on it.  

Models - https://drive.google.com/drive/folders/19q-89TjeyaM2d8KbWdjWOjJV1Ow1ARi2?usp=sharing <br>
Datasets link - 
- https://github.com/GokulNC/NLP-Exercises/tree/master/Transliteration-Indian-Languages/Original-NEWS2012-data
- https://drive.google.com/file/d/1Z6Qxr-q-F54iYB2G1AyoDymBh64f5REZ/view

Credits.
* Transliteration started code for padhAI course.
* Seq2seq model's detail https://arxiv.org/pdf/1507.05717.pdf
* Test images downloaded from https://www.picxy.com/
