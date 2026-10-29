import gradio as gr, joblib, pandas as pd

model = joblib.load("model.joblib")
LABELS = ["Low", "Medium", "High"]
VILLAINS = ["Green Goblin", "Mysterio", "Kingpin", "Doctor Octopus", "Sandman", "Venom", "The Lizard", "Prowler"]
TYPES = ["Robbery", "Hostage Rescue", "Portal Disturbance", "Vehicle Chase", "Public Attack", "Reactor Sabotage", "Lab Break-in"]
PLACES = ["Stark Tower", "Manhattan", "Harlem", "Times Square", "Oscorp Labs", "Queens", "Brooklyn"]

def predict(villain, itype, location, weather, tod, civilians, damage, web, sense, backup, escape, delay):
    row = pd.DataFrame([dict(villain=villain, incident_type=itype, location=location, weather=weather,
                             time_of_day=tod, civilians_at_risk=civilians, property_damage_level=damage,
                             web_trap_strength=web, spider_sense_alert=sense, backup_status=backup,
                             villain_escape_risk=escape, response_delay_minutes=delay)])
    return {l: float(p) for l, p in zip(LABELS, model.predict_proba(row)[0])}

demo = gr.Interface(
    predict,
    [gr.Dropdown(VILLAINS, value=VILLAINS[0], label="Villain"), gr.Dropdown(TYPES, value=TYPES[0], label="Incident type"),
     gr.Dropdown(PLACES, value=PLACES[0], label="Location"), gr.Dropdown(["Clear", "Rain", "Storm", "Fog"], value="Clear", label="Weather"),
     gr.Dropdown(["Morning", "Afternoon", "Evening", "Night"], value="Night", label="Time of day"),
     gr.Slider(0, 40, 15, step=1, label="Civilians at risk"), gr.Slider(1, 10, 5, step=1, label="Property damage level"),
     gr.Slider(1, 10, 5, step=1, label="Web trap strength"), gr.Dropdown(["Low", "Moderate", "High"], value="Moderate", label="Spider-sense alert"),
     gr.Dropdown(["Available", "Delayed", "Unavailable"], value="Available", label="Backup status"),
     gr.Slider(1, 10, 5, step=1, label="Villain escape risk"), gr.Slider(0, 30, 10, step=1, label="Response delay (minutes)")],
    gr.Label(num_top_classes=3, label="Predicted response priority"),
    title="Spider-Sense Triage Console", description="Enter incident details to predict the response priority.")

if __name__ == "__main__":
    demo.launch()
