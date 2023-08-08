import cv2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import Toplevel 


class_labels = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles'] 
class FoodClassifierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Food Classifier")

        self.new_window_button = ttk.Button(root, text="Show pred", command=self.show_predictions)
        self.new_window_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.model = tf.keras.models.load_model('food_model.hdf5')
        self.current_frame = None  

        self.frame_label = ttk.Label(root)
        self.frame_label.pack()

        self.update_frame()

    def classify_food(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = np.expand_dims(resized_frame, axis=0)
        predictions = self.model.predict(resized_frame)[0]
        top3_indices = np.argsort(predictions)[::-1][:3]
        top3_labels = [class_labels[i] for i in top3_indices]
        top3_probs = [predictions[i] for i in top3_indices]
        return top3_labels, top3_probs
        
    def show_predictions(self):
        if self.current_frame is not None:
            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            top3_labels, top3_probs = self.classify_food(frame)

            prediction_window = Toplevel(self.root)
            prediction_window.title("Pred results: ")

            image = cv2.resize(self.current_frame, (400, 300))
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            image_label = ttk.Label(prediction_window, image=photo)
            image_label.image = photo
            image_label.pack()

            predictions_text = "Pred results :\n"
            for label, prob in zip(top3_labels, top3_probs):
                predictions_text += f"{label}: {prob:.2%}\n"
            predictions_label = ttk.Label(prediction_window, text=predictions_text, font=("Helvetica", 12))
            predictions_label.pack()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame  
            image = cv2.resize(frame, (400, 300))
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image=image)
            self.frame_label.config(image=self.photo)
            self.frame_label.image = self.photo
        self.root.after(10, self.update_frame)
root = tk.Tk()
app = FoodClassifierApp(root)
root.mainloop()

app.cap.release()
cv2.destroyAllWindows()