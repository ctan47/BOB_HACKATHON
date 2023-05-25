import tensorflow as tf

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load image for object detection
image_path = 'path_to_your_image.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

# Perform object detection
predictions = model.predict(image_array)
results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

# Display the top 5 predicted objects
for result in results:
    print(f"{result[1]}: {result[2]*100:.2f}%")
