# DinoGameRES — AI Agent for Chrome's Dinosaur Game

An AI that automatically plays Chrome's offline dinosaur game using reinforcement learning (PPO algorithm via Stable-Baselines3) and a rule-based vision script. Two approaches are included: trained neural network and manual code-based control.

## Approaches

### 1. AI (Trained Neural Network)
Uses Stable-Baselines3 (PPO) + Gymnasium environment. The agent captures the game screen, detects obstacles via OpenCV, and learns to jump at the right moment through reinforcement learning.

### 2. Manual (Rule-Based)
A deterministic script that detects obstacles using pixel coordinates on the screen and triggers jumps at fixed thresholds. Fast and reliable, but not adaptive.

## Scripts

| Script | Description |
|--------|-------------|
| `Cerebro.py` | Visualizes what the AI "sees" — game region and detected obstacles |
| `DinoEntorno.py` | Gymnasium environment definition (observation space, actions, reward) |
| `Entrenamiento.py` | Trains the PPO agent and saves the model |
| `JuegaIA.py` | Loads the trained model and plays the game autonomously |

## Requirements

- Python 3.9
- Screen resolution: 1920x1200 (pixel coordinates are hardcoded for this resolution)
- Chrome browser with the dinosaur game open (`chrome://dino`)

## Dependencies

```
stable-baselines3==2.7.0
torch==2.8.0+cpu
gymnasium==1.1.1
opencv-python==4.12.0.88
mss==10.1.0
pynput==1.8.1
pyautogui==0.9.54
numpy==2.0.2
```

## Setup

```bash
git clone https://github.com/EmilioUgarte65/DinoGameRES.git
cd DinoGameRES
pip install stable-baselines3 torch gymnasium opencv-python mss pynput pyautogui numpy
```

## Usage

### Train the agent

```bash
python Entrenamiento.py
```

### Watch the AI play

```bash
python JuegaIA.py
```

### Debug vision

```bash
python Cerebro.py
```

## Notes

- Trained on a 13th Gen Intel Core i7-13700HX (CPU only)
- Screen resolution **must** be 1920x1200 for the pixel-coordinate scripts to work correctly
- Open Chrome's dino game before running any script

## License

MIT
