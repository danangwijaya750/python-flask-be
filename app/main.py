from flask import Flask
  
app = Flask(__name__)
  
@app.route("/summary")
def home_view():
        return "summarization"