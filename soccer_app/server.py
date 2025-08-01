from flask import Flask, render_template, request
from team import team_bp
from player import player_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(team_bp, url_prefix='/team')
app.register_blueprint(player_bp, url_prefix='/player')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)