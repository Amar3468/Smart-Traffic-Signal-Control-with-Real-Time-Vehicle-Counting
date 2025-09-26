from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")   # loads templates/index.html

@app.route("/get_green_time")
def get_green_time():
    try:
        with open("signal_time.txt", "r") as f:
            green_time = int(f.read().strip())
    except:
        green_time = 3   # fallback if file not found
    return jsonify({"duration": green_time})

if __name__ == "__main__":
    app.run(debug=True)
