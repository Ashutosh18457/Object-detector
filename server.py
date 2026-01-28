import pyttsx3  # Text-to-Speech library
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import required classes
import os  # For system commands
import datetime  # For date and time
import speech_recognition as sr  # For voice inputpy
class Ash:
    def __init__(self, model_name="gpt2"):
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Initialize the text-to-speech engine
        self.tts_engine = pyttsx3.init()

    def chat(self, prompt):
        # Tokenize input and generate response
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def speak(self, text):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def execute_command(self, command):
        """Execute specific commands like a virtual assistant."""
        if "time" in command:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            response = f"The current time is {current_time}."
        elif "date" in command:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            response = f"Today's date is {current_date}."
        elif "open notepad" in command:
            os.system("notepad")
            response = "Opening Notepad."
        elif "shutdown" in command:
            response = "Shutting down the system. Goodbye!"
            self.speak(response)
            os.system("shutdown /s /t 1")
        else:
            response = "I'm not sure how to do that. Let me think about it."
        return response

    def listen(self):
        """Listen to the user's voice input."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            self.speak("I'm listening.")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                print(f"You said: {command}")
                return command.lower()
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't catch that. Could you repeat?")
                return None
            except sr.RequestError:
                self.speak("Sorry, there seems to be an issue with the speech recognition service.")
                return None

if __name__ == "__main__":
    print("Welcome to Ash! Say 'exit' to quit.")
    ash = Ash()

    while True:
        user_input = ash.listen()
        if user_input is None:
            continue
        if "exit" in user_input:
            print("Goodbye!")
            ash.speak("Goodbye!")
            break
        elif "ash" in user_input:
            # Handle specific commands
            response = ash.execute_command(user_input)
        else:
            # General chat
            response = ash.chat(user_input)
        print(f"Ash: {response}")
        ash.speak(response)