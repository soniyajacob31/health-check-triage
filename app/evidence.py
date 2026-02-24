"""
evidence.py
============
Generates evidence-based explanations for the triage recommendation
using the ML model's probability outputs and publicly available
reference statistics from CDC NHAMCS and published literature.
No individual patient data or dataset-specific counts are exposed.
"""

import json
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent / "config"

_pub_rates = None


def _load():
    global _pub_rates
    if _pub_rates is None:
        with open(CFG_DIR / "public_reference_rates.json") as f:
            _pub_rates = json.load(f)


def get_evidence(patient_state, prediction):
    """
    Generate evidence for the results page.
    Uses model probabilities + published reference rates.
    """
    _load()

    probas = prediction.get("probabilities", {})
    level1_prob = probas.get(1, 0)

    overall = _pub_rates["overall"]
    by_symptom = _pub_rates["by_symptom"]

    # Collect published rates for the patient's symptoms
    matched_admission = []
    matched_mortality = []
    matched_sources = set()

    for sym_id in patient_state.selected_symptoms:
        sym_rates = by_symptom.get(sym_id)
        if sym_rates:
            matched_admission.append(sym_rates["admission_rate"])
            matched_mortality.append(sym_rates["mortality_rate"])
            matched_sources.add(sym_rates.get("source", ""))

    # Use worst-case rates across matched symptoms (safety-first)
    if matched_admission:
        pub_admission = max(matched_admission)
        pub_mortality = max(matched_mortality)
    else:
        pub_admission = overall["admission_rate"]
        pub_mortality = overall["seven_day_mortality"]

    # Compute the three risk percentages:
    # 1. Immediate attention: model's P(Level 1) as a percentage
    #    This reflects the patient-specific prediction.
    immediate_pct = round(level1_prob * 100, 1)

    # 2. Hospitalization: published admission rate for this symptom type,
    #    weighted by model confidence in Level 1. If model says high risk,
    #    use the higher of model prediction or published rate.
    hosp_pct = round(max(pub_admission, level1_prob * pub_admission * 2), 1)
    hosp_pct = min(hosp_pct, 95.0)

    # 3. Mortality: published mortality rate for this symptom type.
    death_pct = round(pub_mortality, 1)

    # For red-flag overrides, ensure percentages reflect urgency
    if prediction.get("red_flag"):
        immediate_pct = max(immediate_pct, 95.0)
        hosp_pct = max(hosp_pct, pub_admission)

    # 4. Likelihood of something serious: probability the underlying
    #    condition is not benign. Uses P(Level 1) + P(Level 2) — the
    #    chance the patient needs same-day medical care.
    p_serious = (probas.get(1, 0) + probas.get(2, 0)) * 100
    p_serious = round(min(p_serious, 99.0), 1)
    if prediction.get("red_flag"):
        p_serious = max(p_serious, 95.0)

    result = {
        "summary": "",
        "symptom_stats": [],
        "watch_for": [],
        "reassurance": "",
        "risk_pcts": {
            "immediate_attention": immediate_pct,
            "hospitalization": hosp_pct,
            "death": death_pct,
        },
    }

    # Build summary (no patient counts — DUA compliant)
    symptom_names = []
    with open(CFG_DIR / "symptom_categories.json") as f:
        sym_map = {c["id"]: c["label"] for c in json.load(f)}

    for sym_id in patient_state.selected_symptoms:
        if sym_id in sym_map and sym_id != "other":
            symptom_names.append(sym_map[sym_id])

    if symptom_names:
        primary = symptom_names[0]
        result["summary"] = (
            f"Based on published emergency department data, patients "
            f"presenting with symptoms similar to yours (\"{primary}\") "
            f"have an estimated {hosp_pct}% rate of hospital admission."
        )
    else:
        result["summary"] = (
            "This recommendation is based on your responses analyzed "
            "against published emergency department outcome data."
        )

    # Published reference info per symptom
    for sym_id in patient_state.selected_symptoms:
        sym_rates = by_symptom.get(sym_id)
        if sym_rates and sym_id in sym_map:
            result["symptom_stats"].append({
                "label": sym_map[sym_id],
                "admission_rate": sym_rates["admission_rate"],
                "mortality_rate": sym_rates["mortality_rate"],
            })

    # Reassurance statement + symptom-specific watch-outs
    level = prediction.get("level", 3)
    result["reassurance"] = _build_reassurance(
        level, symptom_names, patient_state, p_serious,
        specialist_info=prediction.get("specialist"),
    )

    # Watch-for signs tailored to the patient's symptoms
    result["watch_for"] = _build_watch_for(patient_state.selected_symptoms)

    # "If This Happens, Escalate" statements
    result["escalation"] = _build_escalation(patient_state.selected_symptoms)

    # Triage nurse summary (shown when recommendation is ER/UC/PCP)
    result["triage_summary"] = _build_triage_summary(patient_state, prediction)

    # Home remedies (shown for Level 5 reassurance)
    result["home_remedies"] = _build_home_remedies(
        patient_state.selected_symptoms, level
    )

    # Specialist detail (from Arvig et al. WestJEM 2022 complaint-diagnosis map)
    specialist_info = prediction.get("specialist")
    if specialist_info:
        result["specialist"] = specialist_info

    # Differential diagnosis based on symptoms, PMH, demographics
    result["differential"] = _build_differential(
        patient_state.selected_symptoms, patient_state, level
    )

    return result


# ── Symptom-specific watch-for signs ─────────────────────────────────
SYMPTOM_WATCH_FOR = {
    "chest_pain": [
        "Pain that spreads to your arm, jaw, neck, or back",
        "Chest pain that gets worse with activity or doesn't go away with rest",
        "Chest pain with sweating, nausea, or lightheadedness",
    ],
    "shortness_of_breath": [
        "Breathing that keeps getting harder or faster",
        "Lips or fingertips turning blue or gray",
        "Unable to speak in full sentences due to breathlessness",
    ],
    "abdominal_pain": [
        "Belly pain that becomes severe or constant",
        "Belly that feels hard or very tender to touch",
        "Vomiting blood or passing dark/bloody stool",
    ],
    "headache": [
        "Sudden, severe headache unlike anything you've had before",
        "Headache with stiff neck, high fever, or confusion",
        "Headache with vision changes, weakness, or trouble speaking",
    ],
    "fever": [
        "Temperature above 103\u00b0F (39.4\u00b0C) that won't come down",
        "Fever with a stiff neck, rash, or confusion",
        "Fever with difficulty breathing or chest pain",
    ],
    "dizziness": [
        "Dizziness with slurred speech or weakness on one side",
        "Fainting or passing out",
        "Dizziness with chest pain or irregular heartbeat",
    ],
    "injury_fall": [
        "Increasing swelling, numbness, or inability to move the injured area",
        "Injury with signs of infection (redness spreading, warmth, pus)",
        "Head injury followed by confusion, vomiting, or worsening headache",
    ],
    "back_pain": [
        "Loss of bladder or bowel control",
        "Numbness or weakness spreading to both legs",
        "Severe pain that wakes you from sleep or doesn't improve",
    ],
    "weakness": [
        "Sudden weakness or numbness on one side of your body",
        "Weakness with trouble speaking, confusion, or vision changes",
        "Weakness that keeps getting worse over hours",
    ],
    "nausea_vomiting": [
        "Vomiting that won't stop or contains blood",
        "Signs of dehydration (very dry mouth, no urination, dizziness)",
        "Severe belly pain with vomiting",
    ],
    "rash": [
        "Rash that spreads quickly or comes with fever",
        "Swelling of face, lips, or throat",
        "Blisters or skin that looks infected (warm, red, swelling)",
    ],
    "urinary": [
        "Fever or chills with urinary symptoms",
        "Blood in your urine that is heavy or won't stop",
        "Severe flank or back pain with urinary symptoms",
    ],
    "allergic_reaction": [
        "Swelling of your face, lips, tongue, or throat",
        "Trouble breathing or swallowing",
        "Feeling faint or dizzy after exposure to an allergen",
    ],
    "gi_bleed": [
        "Large amounts of blood in vomit or stool",
        "Feeling lightheaded, dizzy, or like you might pass out",
        "Rapid heartbeat or cold/clammy skin",
    ],
    "anxiety_depression": [
        "Thoughts of hurting yourself or others",
        "Feeling unable to cope or function",
        "Panic symptoms that don't improve (racing heart, can't breathe)",
    ],
    "substance_use": [
        "Confusion, seizures, or loss of consciousness",
        "Difficulty breathing or very slow breathing",
        "Chest pain or irregular heartbeat",
    ],
}

GENERAL_WATCH_FOR = [
    "Sudden or severe chest pain or pressure",
    "Difficulty breathing that gets worse",
    "Sudden confusion, trouble speaking, or weakness on one side",
    "Fainting or loss of consciousness",
    "Severe or worsening pain that doesn't improve",
]


def _build_watch_for(selected_symptoms):
    """Build a tailored watch-for list based on the patient's symptoms."""
    signs = []
    seen = set()
    for sym_id in selected_symptoms:
        for sign in SYMPTOM_WATCH_FOR.get(sym_id, []):
            if sign not in seen:
                signs.append(sign)
                seen.add(sign)

    for sign in GENERAL_WATCH_FOR:
        if sign not in seen:
            signs.append(sign)
            seen.add(sign)

    return signs[:8]


# ── "If This Happens, Escalate" statements ───────────────────────────
SYMPTOM_ESCALATION = {
    "chest_pain": [
        {"if_sign": "your chest pain gets worse or spreads to your arm, jaw, or neck",
         "then_action": "Call 911 immediately", "severity": "critical"},
        {"if_sign": "you start sweating, feel nauseous, or get lightheaded with the chest pain",
         "then_action": "Call 911 immediately", "severity": "critical"},
        {"if_sign": "the pain doesn\u2019t go away after 15 minutes of rest",
         "then_action": "Go to the Emergency Department", "severity": "urgent"},
    ],
    "shortness_of_breath": [
        {"if_sign": "your breathing gets harder, faster, or you can\u2019t speak full sentences",
         "then_action": "Call 911 immediately", "severity": "critical"},
        {"if_sign": "your lips or fingertips turn blue or gray",
         "then_action": "Call 911 immediately", "severity": "critical"},
    ],
    "headache": [
        {"if_sign": "you get a sudden, severe headache that\u2019s the worst of your life",
         "then_action": "Call 911 \u2014 this could be bleeding in the brain", "severity": "critical"},
        {"if_sign": "you develop a stiff neck with fever",
         "then_action": "Go to the Emergency Department now", "severity": "critical"},
        {"if_sign": "you notice weakness on one side, trouble speaking, or vision changes",
         "then_action": "Call 911 \u2014 these could be signs of a stroke", "severity": "critical"},
    ],
    "abdominal_pain": [
        {"if_sign": "your belly pain becomes severe or constant and won\u2019t go away",
         "then_action": "Go to the Emergency Department", "severity": "urgent"},
        {"if_sign": "you see blood in your vomit or stool",
         "then_action": "Go to the Emergency Department immediately", "severity": "critical"},
        {"if_sign": "your belly becomes hard, swollen, or very tender to touch",
         "then_action": "Go to the Emergency Department", "severity": "urgent"},
    ],
    "fever": [
        {"if_sign": "your temperature goes above 103\u00b0F (39.4\u00b0C) and won\u2019t come down",
         "then_action": "Go to the Emergency Department", "severity": "urgent"},
        {"if_sign": "you develop a fever with confusion, stiff neck, or rash",
         "then_action": "Call 911 or go to the ER immediately", "severity": "critical"},
    ],
    "dizziness": [
        {"if_sign": "you develop slurred speech or weakness on one side of your body",
         "then_action": "Call 911 \u2014 these are signs of a stroke", "severity": "critical"},
        {"if_sign": "you faint or pass out",
         "then_action": "Call 911 or go to the Emergency Department", "severity": "critical"},
    ],
    "back_pain": [
        {"if_sign": "you lose control of your bladder or bowels",
         "then_action": "Go to the Emergency Department immediately", "severity": "critical"},
        {"if_sign": "numbness or weakness spreads to both legs",
         "then_action": "Go to the Emergency Department now", "severity": "critical"},
    ],
    "weakness": [
        {"if_sign": "you develop sudden weakness or numbness on one side of your body",
         "then_action": "Call 911 \u2014 this could be a stroke", "severity": "critical"},
    ],
    "nausea_vomiting": [
        {"if_sign": "you can\u2019t keep any fluids down for more than 12 hours",
         "then_action": "Go to an Urgent Care or Emergency Department for fluids", "severity": "urgent"},
        {"if_sign": "you see blood in your vomit",
         "then_action": "Go to the Emergency Department immediately", "severity": "critical"},
    ],
    "rash": [
        {"if_sign": "the rash spreads quickly and comes with fever or trouble breathing",
         "then_action": "Call 911 or go to the Emergency Department", "severity": "critical"},
    ],
    "urinary": [
        {"if_sign": "you develop fever, chills, or severe flank pain with urinary symptoms",
         "then_action": "Go to the Emergency Department", "severity": "urgent"},
    ],
    "allergic_reaction": [
        {"if_sign": "your face, lips, or throat start to swell, or you have trouble breathing",
         "then_action": "Use your EpiPen if you have one and call 911 immediately", "severity": "critical"},
    ],
    "gi_bleed": [
        {"if_sign": "you feel lightheaded, dizzy, or like you might pass out",
         "then_action": "Call 911 immediately", "severity": "critical"},
    ],
    "injury_fall": [
        {"if_sign": "after a head injury you develop confusion, vomiting, or worsening headache",
         "then_action": "Go to the Emergency Department immediately", "severity": "critical"},
        {"if_sign": "you notice increasing swelling, numbness, or can\u2019t move the injured area",
         "then_action": "Go to an Urgent Care or Emergency Department", "severity": "urgent"},
    ],
    "anxiety_depression": [
        {"if_sign": "you have thoughts of hurting yourself or others",
         "then_action": "Call 988 (Suicide & Crisis Lifeline) or go to the nearest ER", "severity": "critical"},
    ],
}

GENERAL_ESCALATION = [
    {"if_sign": "your symptoms suddenly get much worse",
     "then_action": "Call 911 or go to the nearest Emergency Department", "severity": "critical"},
    {"if_sign": "you develop new trouble breathing, chest pain, or confusion",
     "then_action": "Call 911 immediately", "severity": "critical"},
    {"if_sign": "your symptoms don\u2019t improve after 24\u201348 hours or keep getting worse",
     "then_action": "See a doctor sooner than planned or go to Urgent Care", "severity": "watch"},
]


def _build_escalation(selected_symptoms):
    """Build a list of 'If X → Then Y' escalation statements."""
    items = []
    seen_actions = set()
    for sym_id in selected_symptoms:
        for esc in SYMPTOM_ESCALATION.get(sym_id, []):
            key = esc["then_action"]
            if key not in seen_actions:
                items.append(esc)
                seen_actions.add(key)

    for esc in GENERAL_ESCALATION:
        if esc["then_action"] not in seen_actions:
            items.append(esc)
            seen_actions.add(esc["then_action"])

    return items[:8]


# ── Triage nurse summary builder ─────────────────────────────────────
CONCERNING_ANSWER_VALUES = {
    "severe", "worst", "sudden", "thunderclap", "yes", "vomiting",
    "projectile", "pressure", "tightness", "tearing", "left_arm",
    "jaw_neck", "multiple", "one_side_face", "one_side_body",
    "loss", "double", "heart_attack", "heart_disease", "stent_surgery",
    "rapid", "yes_today", "yes_recently", "at_rest",
}


def _build_triage_summary(patient_state, prediction):
    """Build a structured summary the patient can show the triage nurse."""
    items = []

    if patient_state.age and patient_state.sex:
        items.append(f"{patient_state.age}-year-old {patient_state.sex}")

    if patient_state.symptom_text:
        items.append(f"Came in for: {patient_state.symptom_text}")

    for entry in patient_state.interview_history:
        qid = entry.get("question_id", "")
        answer = entry.get("answer", "")
        if "__" in qid and answer in CONCERNING_ANSWER_VALUES:
            items.append(
                f"{entry['question_text']}: {entry['answer_display']}"
            )

    if patient_state.pmh:
        items.append(f"Medical history: {', '.join(patient_state.pmh)}")
    elif (
        patient_state.pmh_text
        and patient_state.pmh_text.lower() not in ("none", "no", "nothing", "n/a")
    ):
        items.append(f"Medical history: {patient_state.pmh_text}")

    if prediction.get("red_flag"):
        items.append(f"\u26a0\ufe0f Flagged: {prediction['red_flag']['name']}")

    return items


# ── Home remedies (for reassurance-level recommendations) ────────────
SYMPTOM_HOME_REMEDIES = {
    "headache": [
        {"remedy": "Rest in a quiet, dark room", "detail": "Reducing light and noise can help a headache ease."},
        {"remedy": "Stay hydrated", "detail": "Drink water or clear fluids \u2014 dehydration is a common headache trigger."},
        {"remedy": "Over-the-counter pain relief", "detail": "Acetaminophen (Tylenol) or ibuprofen (Advil) as directed on the label."},
        {"remedy": "Cold compress", "detail": "Place a cold cloth or ice pack on your forehead for 15 minutes."},
        {"remedy": "Limit screen time", "detail": "Take a break from phones, computers, and bright screens."},
    ],
    "fever": [
        {"remedy": "Rest and sleep", "detail": "Your body fights illness best when resting."},
        {"remedy": "Stay hydrated", "detail": "Drink water, broth, or electrolyte drinks frequently."},
        {"remedy": "Over-the-counter fever reducer", "detail": "Acetaminophen or ibuprofen as directed. Do NOT give aspirin to children."},
        {"remedy": "Lukewarm compress", "detail": "A lukewarm (not cold) washcloth on the forehead can help."},
        {"remedy": "Dress lightly", "detail": "Wear light clothing and use a light blanket."},
    ],
    "nausea_vomiting": [
        {"remedy": "Sip clear fluids slowly", "detail": "Small sips of water, ginger ale, or electrolyte drinks every few minutes."},
        {"remedy": "Try the BRAT diet", "detail": "Bananas, rice, applesauce, and toast are gentle on the stomach."},
        {"remedy": "Ginger", "detail": "Ginger tea, ginger ale, or ginger candies can help settle nausea."},
        {"remedy": "Avoid triggers", "detail": "Stay away from greasy, spicy, or strong-smelling foods."},
        {"remedy": "Rest sitting up", "detail": "Lying flat can make nausea worse \u2014 try propping yourself up."},
    ],
    "cough": [
        {"remedy": "Warm fluids", "detail": "Tea with honey, warm water with lemon, or broth can soothe your throat."},
        {"remedy": "Honey", "detail": "A spoonful of honey can help calm a cough (not for children under 1)."},
        {"remedy": "Humidifier", "detail": "Adding moisture to the air can help loosen congestion."},
        {"remedy": "Cough drops or lozenges", "detail": "Over-the-counter throat lozenges can provide temporary relief."},
        {"remedy": "Elevate your head", "detail": "Use an extra pillow when sleeping to reduce nighttime coughing."},
    ],
    "sore_throat": [
        {"remedy": "Warm saltwater gargle", "detail": "Mix \u00bd teaspoon salt in warm water and gargle for 30 seconds."},
        {"remedy": "Warm fluids", "detail": "Tea with honey, warm broth, or warm water with lemon."},
        {"remedy": "Throat lozenges", "detail": "Over-the-counter lozenges or hard candy to keep your throat moist."},
        {"remedy": "Over-the-counter pain relief", "detail": "Acetaminophen or ibuprofen can reduce pain and swelling."},
        {"remedy": "Rest your voice", "detail": "Try to talk less and avoid whispering (it strains your throat more)."},
    ],
    "back_pain": [
        {"remedy": "Gentle movement", "detail": "Avoid bed rest \u2014 gentle walking and stretching help more than staying still."},
        {"remedy": "Alternate ice and heat", "detail": "Ice for the first 48 hours (20 min on/off), then switch to heat."},
        {"remedy": "Over-the-counter anti-inflammatory", "detail": "Ibuprofen (Advil) or naproxen (Aleve) as directed on the label."},
        {"remedy": "Avoid heavy lifting", "detail": "Don\u2019t lift anything heavy until the pain improves."},
        {"remedy": "Good posture", "detail": "Sit and stand straight \u2014 slouching puts extra strain on your back."},
    ],
    "extremity_pain": [
        {"remedy": "RICE method", "detail": "Rest, Ice (20 min on/off), Compression (wrap), Elevation (raise it up)."},
        {"remedy": "Over-the-counter anti-inflammatory", "detail": "Ibuprofen or naproxen to reduce pain and swelling."},
        {"remedy": "Gentle stretching", "detail": "Light stretching may help if the pain is from a muscle strain."},
        {"remedy": "Avoid overuse", "detail": "Give the area time to heal \u2014 don\u2019t push through the pain."},
    ],
    "rash": [
        {"remedy": "Cool compress", "detail": "A cool, damp cloth on the rash can relieve itching and swelling."},
        {"remedy": "Over-the-counter antihistamine", "detail": "Benadryl (diphenhydramine) or Zyrtec (cetirizine) for itching."},
        {"remedy": "Moisturize", "detail": "Unscented lotion or aloe vera gel can soothe irritated skin."},
        {"remedy": "Avoid scratching", "detail": "Scratching can make it worse or cause infection."},
        {"remedy": "Avoid irritants", "detail": "Stay away from harsh soaps, new detergents, or anything that may have triggered it."},
    ],
    "dizziness": [
        {"remedy": "Sit or lie down right away", "detail": "This prevents falls and helps blood flow to your brain."},
        {"remedy": "Stay hydrated", "detail": "Dizziness is often caused by dehydration \u2014 drink water or electrolyte drinks."},
        {"remedy": "Eat something", "detail": "Low blood sugar can cause dizziness \u2014 try a light snack."},
        {"remedy": "Avoid sudden movements", "detail": "Get up slowly from sitting or lying down."},
        {"remedy": "Rest", "detail": "Give your body time to recover in a comfortable position."},
    ],
    "abdominal_pain": [
        {"remedy": "Clear liquids", "detail": "Start with water, broth, or ginger tea. Avoid solid food until the pain eases."},
        {"remedy": "Warm compress", "detail": "A heating pad or warm towel on your belly can help relax cramps."},
        {"remedy": "Rest", "detail": "Lie in a comfortable position \u2014 try lying on your side with knees pulled up."},
        {"remedy": "Avoid trigger foods", "detail": "Stay away from fatty, spicy, or acidic foods until you feel better."},
        {"remedy": "Peppermint tea", "detail": "Peppermint can help relax stomach muscles and ease bloating."},
    ],
    "urinary": [
        {"remedy": "Drink plenty of water", "detail": "Flushing your system with water can help mild urinary symptoms."},
        {"remedy": "Cranberry juice", "detail": "Unsweetened cranberry juice may help prevent bacteria from sticking."},
        {"remedy": "Avoid caffeine and alcohol", "detail": "These can irritate the bladder and make symptoms worse."},
        {"remedy": "Warm compress", "detail": "A warm cloth on your lower belly can ease discomfort."},
    ],
    "anxiety_depression": [
        {"remedy": "Deep breathing", "detail": "Breathe in for 4 seconds, hold for 4, out for 4. Repeat 5 times."},
        {"remedy": "Grounding technique", "detail": "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste."},
        {"remedy": "Talk to someone", "detail": "Call a friend, family member, or the 988 Suicide & Crisis Lifeline."},
        {"remedy": "Physical activity", "detail": "Even a 10-minute walk can help reduce anxiety."},
        {"remedy": "Limit caffeine", "detail": "Coffee and energy drinks can worsen anxiety symptoms."},
    ],
    "insomnia": [
        {"remedy": "Keep a regular sleep schedule", "detail": "Go to bed and wake up at the same time each day."},
        {"remedy": "Avoid screens before bed", "detail": "Put away phones and laptops at least 30 minutes before sleep."},
        {"remedy": "Create a calm environment", "detail": "Cool, dark, quiet room. Try earplugs or a white noise machine."},
        {"remedy": "Avoid caffeine after noon", "detail": "Caffeine stays in your system for hours and disrupts sleep."},
    ],
}

GENERAL_HOME_REMEDIES = [
    {"remedy": "Rest", "detail": "Give your body time to heal \u2014 take it easy for a day or two."},
    {"remedy": "Stay hydrated", "detail": "Drink water, herbal tea, or clear broth throughout the day."},
    {"remedy": "Over-the-counter pain relief", "detail": "Acetaminophen (Tylenol) or ibuprofen (Advil) as directed if needed."},
    {"remedy": "Monitor your symptoms", "detail": "Keep track of how you feel. If things get worse, seek medical care."},
]


def _build_home_remedies(selected_symptoms, level):
    """Build a list of home remedies for reassurance-level recommendations."""
    if level > 5:
        return []

    remedies = []
    seen = set()
    for sym_id in selected_symptoms:
        for rem in SYMPTOM_HOME_REMEDIES.get(sym_id, []):
            if rem["remedy"] not in seen:
                remedies.append(rem)
                seen.add(rem["remedy"])

    if not remedies:
        for rem in GENERAL_HOME_REMEDIES:
            if rem["remedy"] not in seen:
                remedies.append(rem)
                seen.add(rem["remedy"])

    return remedies[:6]


# ── Reassurance statement builder ────────────────────────────────────
def _build_reassurance(level, symptom_names, patient_state, p_serious,
                       specialist_info=None):
    """Generate a warm, personalized reassurance paragraph."""
    symptom_desc = symptom_names[0] if symptom_names else "your symptoms"
    age = patient_state.age or "unknown"

    if level == 1:
        return (
            f"We understand that {symptom_desc.lower()} can be frightening. "
            f"Based on your responses, this combination of symptoms warrants "
            f"prompt medical evaluation to rule out anything serious. "
            f"Many patients who go to the emergency department for similar "
            f"symptoms are treated and sent home the same day. Going to the "
            f"ER is the safe and right thing to do \u2014 it does not mean the "
            f"worst is happening. Getting checked out gives you and your "
            f"doctors the information needed to take care of you properly."
        )
    elif level == 2:
        return (
            f"Your symptoms suggest you should be seen by a healthcare "
            f"provider today, but the situation is unlikely to be "
            f"life-threatening. An urgent care center can evaluate you, "
            f"run basic tests, and provide treatment or refer you if needed. "
            f"Most people with similar symptoms are treated and feel better "
            f"within a short time. If at any point your symptoms get "
            f"significantly worse, go to the nearest emergency department."
        )
    elif level == 3:
        sp_info = specialist_info or {}
        if sp_info.get("pcp_first") and sp_info.get("specialist"):
            sp_name = sp_info["specialist"]
            return (
                f"Based on what you've told us, your symptoms are worth "
                f"getting checked, but they don't appear to need emergency "
                f"care right now. We recommend making an appointment with "
                f"your primary care doctor in the next day or two. Your "
                f"doctor can do a thorough evaluation and, if needed, refer "
                f"you to a {sp_name} for further workup. Most conditions "
                f"like this can be effectively managed starting with your "
                f"primary care provider. In the meantime, rest, stay "
                f"hydrated, and monitor how you're feeling."
            )
        return (
            f"Based on what you've told us, your symptoms are concerning "
            f"enough to see a doctor, but they don't appear to need "
            f"emergency care right now. We recommend making an appointment "
            f"with your primary care doctor in the next day or two. "
            f"Your doctor can do a thorough evaluation and order any tests "
            f"that might be helpful. In the meantime, rest, stay hydrated, "
            f"and monitor how you're feeling."
        )
    elif level == 4:
        sp_name = (specialist_info or {}).get("specialist", "a specialist")
        sp_secondary = (specialist_info or {}).get("secondary")
        sp_line = f"a {sp_name}" if sp_name else "a specialist"
        secondary_line = ""
        if sp_secondary:
            secondary_line = (
                f" In some cases, a {sp_secondary} may also be helpful, "
                f"and your doctor can advise which is best for you."
            )
        return (
            f"Your symptoms suggest a condition that may benefit from "
            f"specialized care. Based on published data from large emergency "
            f"department studies, patients with similar complaints most often "
            f"receive a diagnosis managed by {sp_line}. This is not an "
            f"emergency, but a specialist can help you get the right "
            f"diagnosis and treatment plan.{secondary_line} Ask your primary "
            f"care doctor for a referral, or contact the specialist's "
            f"office directly. Most conditions like this respond well to "
            f"treatment once properly identified."
        )
    else:
        return (
            f"The good news is that based on your responses, your symptoms "
            f"are very likely not serious. Many people experience similar "
            f"symptoms that resolve on their own with rest, hydration, and "
            f"over-the-counter remedies. That said, your body knows best "
            f"\u2014 if something feels wrong or your symptoms change, don't "
            f"hesitate to seek medical care. Trust your instincts."
        )


# ── Differential Diagnosis Builder ────────────────────────────────────
# Maps symptoms to specific plain-language diagnoses with approximate
# likelihood based on Arvig et al. WestJEM 2022 discharge diagnosis data
# and standard clinical teaching.

SYMPTOM_DIFFERENTIALS = {
    "chest_pain": [
        {"diagnosis": "Musculoskeletal chest wall pain", "likelihood": "Common", "notes": "Reproducible with palpation; often positional"},
        {"diagnosis": "Gastroesophageal reflux (GERD)", "likelihood": "Common", "notes": "Burning quality; worse after eating or lying down"},
        {"diagnosis": "Anxiety / Panic attack", "likelihood": "Common", "notes": "Palpitations, tingling, sense of doom; often in younger patients"},
        {"diagnosis": "Acute coronary syndrome (heart attack)", "likelihood": "Less common", "notes": "Pressure/squeezing; radiates to arm/jaw; sweating; risk increases with age, diabetes, smoking"},
        {"diagnosis": "Pulmonary embolism", "likelihood": "Less common", "notes": "Sudden onset with shortness of breath; pleuritic; risk with immobility, birth control, recent surgery"},
        {"diagnosis": "Pneumonia / Pleuritis", "likelihood": "Less common", "notes": "Sharp pain with breathing; cough; fever"},
        {"diagnosis": "Pericarditis", "likelihood": "Uncommon", "notes": "Sharp pain worse with lying flat, better leaning forward; recent viral illness"},
        {"diagnosis": "Aortic dissection", "likelihood": "Rare", "notes": "Tearing pain radiating to back; hypertension; medical emergency"},
    ],
    "shortness_of_breath": [
        {"diagnosis": "Asthma / Reactive airway", "likelihood": "Common", "notes": "Wheezing; triggered by allergens, exercise, or cold air"},
        {"diagnosis": "COPD exacerbation", "likelihood": "Common", "notes": "Chronic smoker; worsening baseline dyspnea; productive cough"},
        {"diagnosis": "Pneumonia", "likelihood": "Common", "notes": "Cough, fever, abnormal breath sounds"},
        {"diagnosis": "Heart failure exacerbation", "likelihood": "Less common", "notes": "Leg swelling, orthopnea, history of heart disease"},
        {"diagnosis": "Anxiety / Hyperventilation", "likelihood": "Common", "notes": "Tingling, lightheadedness; often younger patients"},
        {"diagnosis": "Pulmonary embolism", "likelihood": "Less common", "notes": "Sudden onset; pleuritic chest pain; tachycardia"},
        {"diagnosis": "Anemia", "likelihood": "Less common", "notes": "Gradual onset; fatigue; pallor"},
    ],
    "headache": [
        {"diagnosis": "Tension headache", "likelihood": "Very common", "notes": "Band-like pressure; bilateral; associated with stress"},
        {"diagnosis": "Migraine", "likelihood": "Common", "notes": "Unilateral, throbbing; nausea; light/sound sensitivity; aura possible"},
        {"diagnosis": "Sinusitis", "likelihood": "Common", "notes": "Facial pressure; nasal congestion; worse leaning forward"},
        {"diagnosis": "Medication overuse headache", "likelihood": "Less common", "notes": "Chronic daily headache in someone using analgesics >15 days/month"},
        {"diagnosis": "Hypertensive headache", "likelihood": "Less common", "notes": "Severe headache with very high blood pressure"},
        {"diagnosis": "Subarachnoid hemorrhage", "likelihood": "Rare", "notes": "Sudden thunderclap worst-ever headache; medical emergency"},
        {"diagnosis": "Meningitis", "likelihood": "Rare", "notes": "Headache with fever, stiff neck, photophobia"},
        {"diagnosis": "Intracranial mass", "likelihood": "Rare", "notes": "Progressive headache; worse in morning; neurologic deficits"},
    ],
    "abdominal_pain": [
        {"diagnosis": "Gastritis / Dyspepsia", "likelihood": "Very common", "notes": "Epigastric burning; related to meals, NSAID use, or stress"},
        {"diagnosis": "Gastroenteritis", "likelihood": "Common", "notes": "Crampy pain with nausea, vomiting, or diarrhea; often viral"},
        {"diagnosis": "Constipation", "likelihood": "Common", "notes": "Diffuse cramping; infrequent or hard stools"},
        {"diagnosis": "Urinary tract infection", "likelihood": "Common", "notes": "Lower abdominal pain with urinary frequency, burning"},
        {"diagnosis": "Appendicitis", "likelihood": "Less common", "notes": "Starts around navel, migrates to right lower quadrant; fever"},
        {"diagnosis": "Cholecystitis (gallbladder)", "likelihood": "Less common", "notes": "Right upper quadrant pain after fatty meals; nausea"},
        {"diagnosis": "Kidney stones", "likelihood": "Less common", "notes": "Severe flank pain radiating to groin; comes in waves"},
        {"diagnosis": "Small bowel obstruction", "likelihood": "Uncommon", "notes": "Crampy pain, vomiting, distension; history of prior surgery"},
        {"diagnosis": "Ectopic pregnancy", "likelihood": "Uncommon", "notes": "Lower abdominal pain in reproductive-age female; missed period"},
    ],
    "fever": [
        {"diagnosis": "Viral upper respiratory infection", "likelihood": "Very common", "notes": "Cough, congestion, sore throat; self-limited"},
        {"diagnosis": "Urinary tract infection", "likelihood": "Common", "notes": "Fever with urinary symptoms; flank pain if pyelonephritis"},
        {"diagnosis": "Influenza", "likelihood": "Common", "notes": "High fever, body aches, fatigue; seasonal"},
        {"diagnosis": "Pneumonia", "likelihood": "Less common", "notes": "Fever with productive cough, shortness of breath"},
        {"diagnosis": "Cellulitis / Skin infection", "likelihood": "Less common", "notes": "Fever with localized redness, swelling, warmth"},
        {"diagnosis": "COVID-19", "likelihood": "Less common", "notes": "Fever with cough, loss of taste/smell, fatigue"},
        {"diagnosis": "Sepsis", "likelihood": "Uncommon", "notes": "High fever with confusion, rapid breathing, tachycardia; emergency"},
    ],
    "dizziness": [
        {"diagnosis": "Benign positional vertigo (BPPV)", "likelihood": "Very common", "notes": "Brief spinning triggered by head position changes"},
        {"diagnosis": "Orthostatic hypotension", "likelihood": "Common", "notes": "Lightheadedness on standing; dehydration or medication side effect"},
        {"diagnosis": "Vestibular neuritis / Labyrinthitis", "likelihood": "Common", "notes": "Prolonged vertigo after viral illness; may have hearing changes"},
        {"diagnosis": "Anemia", "likelihood": "Less common", "notes": "Gradual onset; fatigue, pallor, shortness of breath"},
        {"diagnosis": "Cardiac arrhythmia", "likelihood": "Less common", "notes": "Intermittent lightheadedness with palpitations"},
        {"diagnosis": "Stroke / TIA", "likelihood": "Uncommon", "notes": "Dizziness with focal weakness, speech difficulty, vision changes; emergency"},
    ],
    "back_pain": [
        {"diagnosis": "Muscle strain / Mechanical back pain", "likelihood": "Very common", "notes": "Related to lifting, activity, or posture; improves with rest"},
        {"diagnosis": "Degenerative disc disease", "likelihood": "Common", "notes": "Chronic; worsens with activity; common in older adults"},
        {"diagnosis": "Herniated disc / Sciatica", "likelihood": "Less common", "notes": "Pain radiating down the leg; numbness or tingling"},
        {"diagnosis": "Spinal stenosis", "likelihood": "Less common", "notes": "Pain with walking, relieved by sitting; older adults"},
        {"diagnosis": "Kidney stones / Pyelonephritis", "likelihood": "Less common", "notes": "Flank pain; may have urinary symptoms or fever"},
        {"diagnosis": "Vertebral compression fracture", "likelihood": "Uncommon", "notes": "Sudden pain after minor trauma; osteoporosis risk"},
        {"diagnosis": "Cauda equina syndrome", "likelihood": "Rare", "notes": "Saddle numbness, bladder/bowel changes, bilateral leg weakness; emergency"},
    ],
    "nausea_vomiting": [
        {"diagnosis": "Gastroenteritis (stomach bug)", "likelihood": "Very common", "notes": "Viral or food-related; diarrhea often accompanies"},
        {"diagnosis": "Food poisoning", "likelihood": "Common", "notes": "Acute onset hours after eating; shared with others who ate same food"},
        {"diagnosis": "Medication side effect", "likelihood": "Common", "notes": "New medication or change in dosage"},
        {"diagnosis": "Gastritis / Peptic ulcer", "likelihood": "Less common", "notes": "Epigastric pain; NSAID or alcohol use"},
        {"diagnosis": "Pregnancy", "likelihood": "Less common", "notes": "Morning nausea in reproductive-age female; missed period"},
        {"diagnosis": "Bowel obstruction", "likelihood": "Uncommon", "notes": "Severe vomiting with abdominal distension and no bowel movements"},
        {"diagnosis": "Pancreatitis", "likelihood": "Uncommon", "notes": "Severe epigastric pain radiating to back; alcohol or gallstone history"},
    ],
    "sore_throat": [
        {"diagnosis": "Viral pharyngitis", "likelihood": "Very common", "notes": "Cough, congestion, runny nose; gradual onset"},
        {"diagnosis": "Streptococcal pharyngitis (strep)", "likelihood": "Common", "notes": "Sudden onset; fever, swollen tonsils, no cough; rapid test available"},
        {"diagnosis": "Infectious mononucleosis", "likelihood": "Less common", "notes": "Fatigue, swollen lymph nodes, possible splenomegaly; young adults"},
        {"diagnosis": "Peritonsillar abscess", "likelihood": "Uncommon", "notes": "Severe unilateral pain, trismus, muffled voice; requires drainage"},
    ],
    "cough": [
        {"diagnosis": "Viral upper respiratory infection", "likelihood": "Very common", "notes": "Self-limited; congestion, sore throat"},
        {"diagnosis": "Acute bronchitis", "likelihood": "Common", "notes": "Persistent cough 1-3 weeks; may produce sputum"},
        {"diagnosis": "Asthma", "likelihood": "Common", "notes": "Cough worse at night or with exercise; wheezing"},
        {"diagnosis": "Post-nasal drip", "likelihood": "Common", "notes": "Throat clearing; sensation of drainage; allergies or sinusitis"},
        {"diagnosis": "Pneumonia", "likelihood": "Less common", "notes": "Cough with fever, shortness of breath, and abnormal breath sounds"},
        {"diagnosis": "GERD-related cough", "likelihood": "Less common", "notes": "Chronic cough; heartburn; worse lying down"},
    ],
    "rash": [
        {"diagnosis": "Contact dermatitis", "likelihood": "Common", "notes": "Itchy rash after exposure to irritant or allergen"},
        {"diagnosis": "Eczema (atopic dermatitis)", "likelihood": "Common", "notes": "Dry, itchy patches; often in skin folds; chronic/recurring"},
        {"diagnosis": "Urticaria (hives)", "likelihood": "Common", "notes": "Raised, itchy welts; allergic trigger; comes and goes"},
        {"diagnosis": "Cellulitis", "likelihood": "Less common", "notes": "Expanding redness, warmth, pain; may have fever; bacterial infection"},
        {"diagnosis": "Shingles (herpes zoster)", "likelihood": "Less common", "notes": "Painful, blistering rash in a band/stripe; one side only"},
        {"diagnosis": "Drug reaction", "likelihood": "Less common", "notes": "Rash after starting new medication"},
    ],
    "urinary": [
        {"diagnosis": "Urinary tract infection (UTI)", "likelihood": "Very common", "notes": "Burning, frequency, urgency; more common in women"},
        {"diagnosis": "Kidney stones", "likelihood": "Less common", "notes": "Severe flank pain with blood in urine"},
        {"diagnosis": "Pyelonephritis (kidney infection)", "likelihood": "Less common", "notes": "UTI symptoms plus fever, flank pain, nausea"},
        {"diagnosis": "Benign prostatic hyperplasia", "likelihood": "Less common", "notes": "Weak stream, frequency, nocturia; older males"},
        {"diagnosis": "Sexually transmitted infection", "likelihood": "Less common", "notes": "Dysuria with discharge; recent sexual exposure"},
    ],
    "extremity_pain": [
        {"diagnosis": "Musculoskeletal strain / Sprain", "likelihood": "Very common", "notes": "Related to activity or injury; localized pain and swelling"},
        {"diagnosis": "Osteoarthritis", "likelihood": "Common", "notes": "Joint pain worse with use; stiffness in morning; older adults"},
        {"diagnosis": "Tendinitis / Bursitis", "likelihood": "Common", "notes": "Pain around a joint with repetitive use"},
        {"diagnosis": "Gout", "likelihood": "Less common", "notes": "Sudden severe joint pain (often big toe); redness, swelling"},
        {"diagnosis": "Fracture", "likelihood": "Less common", "notes": "Pain after trauma; swelling, deformity, inability to bear weight"},
        {"diagnosis": "Deep vein thrombosis (DVT)", "likelihood": "Uncommon", "notes": "Calf pain and swelling; risk with immobility, travel, birth control"},
    ],
    "swelling": [
        {"diagnosis": "Dependent edema", "likelihood": "Common", "notes": "Ankle/leg swelling worse at end of day; improves with elevation"},
        {"diagnosis": "Venous insufficiency", "likelihood": "Common", "notes": "Chronic leg swelling with skin changes; varicose veins"},
        {"diagnosis": "Heart failure", "likelihood": "Less common", "notes": "Bilateral leg swelling with shortness of breath; cardiac history"},
        {"diagnosis": "Deep vein thrombosis", "likelihood": "Less common", "notes": "Unilateral leg swelling with calf pain; acute onset"},
        {"diagnosis": "Cellulitis", "likelihood": "Less common", "notes": "Swelling with redness, warmth, and pain; may have fever"},
    ],
    "eye_problem": [
        {"diagnosis": "Conjunctivitis (pink eye)", "likelihood": "Common", "notes": "Red eye with discharge; itching (allergic) or crusting (infectious)"},
        {"diagnosis": "Dry eye syndrome", "likelihood": "Common", "notes": "Gritty sensation; burning; worse with screen use"},
        {"diagnosis": "Corneal abrasion", "likelihood": "Less common", "notes": "Sharp pain, tearing, light sensitivity after trauma or contact lens"},
        {"diagnosis": "Migraine with visual aura", "likelihood": "Less common", "notes": "Visual changes (zigzag lines, spots) followed by headache"},
        {"diagnosis": "Acute glaucoma", "likelihood": "Rare", "notes": "Severe eye pain, blurred vision, halos; nausea; emergency"},
        {"diagnosis": "Retinal detachment", "likelihood": "Rare", "notes": "Flashes, floaters, curtain over vision; emergency"},
    ],
    "injury_fall": [
        {"diagnosis": "Soft tissue contusion / Bruise", "likelihood": "Very common", "notes": "Pain and bruising at impact site; no fracture"},
        {"diagnosis": "Fracture", "likelihood": "Common", "notes": "Severe pain, swelling, deformity; inability to bear weight or use limb"},
        {"diagnosis": "Sprain / Ligament injury", "likelihood": "Common", "notes": "Joint pain and instability; swelling around joint"},
        {"diagnosis": "Concussion", "likelihood": "Less common", "notes": "Head injury with headache, dizziness, confusion; may have brief LOC"},
        {"diagnosis": "Intracranial hemorrhage", "likelihood": "Rare", "notes": "Head injury with worsening headache, vomiting, altered consciousness; emergency"},
    ],
    "fracture": [
        {"diagnosis": "Simple fracture", "likelihood": "Common", "notes": "Localized pain, swelling, deformity after trauma"},
        {"diagnosis": "Stress fracture", "likelihood": "Less common", "notes": "Gradual onset pain with repetitive activity; common in runners"},
        {"diagnosis": "Pathologic fracture", "likelihood": "Uncommon", "notes": "Fracture from minimal trauma; may indicate osteoporosis or underlying disease"},
    ],
    "pelvic_pain": [
        {"diagnosis": "Menstrual cramps (dysmenorrhea)", "likelihood": "Common", "notes": "Cyclic lower abdominal pain; related to period"},
        {"diagnosis": "Urinary tract infection", "likelihood": "Common", "notes": "Suprapubic pain with urinary symptoms"},
        {"diagnosis": "Ovarian cyst", "likelihood": "Less common", "notes": "Unilateral pelvic pain; may be sudden if ruptured"},
        {"diagnosis": "Pelvic inflammatory disease", "likelihood": "Less common", "notes": "Lower abdominal pain, discharge, fever; sexually active females"},
        {"diagnosis": "Ectopic pregnancy", "likelihood": "Uncommon", "notes": "Pelvic pain with missed period; vaginal bleeding; emergency if ruptured"},
        {"diagnosis": "Kidney stones", "likelihood": "Less common", "notes": "Flank-to-groin pain; hematuria; comes in waves"},
    ],
}

LIKELIHOOD_ORDER = {"Very common": 1, "Common": 2, "Less common": 3, "Uncommon": 4, "Rare": 5}


_SERIOUS_MARKERS = {
    "emergency", "medical emergency", "requires", "drainage",
    "heart attack", "stroke", "tia", "dissection", "embolism",
    "hemorrhage", "meningitis", "sepsis", "obstruction", "ectopic",
    "detachment", "glaucoma", "cauda equina", "intracranial",
    "fracture", "concussion", "appendicitis", "cholecystitis",
    "pyelonephritis", "pancreatitis", "heart failure",
}


def _acuity_score(dx, level):
    """Lower score = more relevant to the recommendation level.

    For ER/UC-level recommendations, serious diagnoses that need to be ruled
    out are the ones *driving* the recommendation, so they rank first.
    For lower-acuity levels, common benign diagnoses rank first.
    """
    text = (dx["diagnosis"] + " " + dx["notes"]).lower()
    is_serious = any(m in text for m in _SERIOUS_MARKERS)

    if level <= 2:
        return 0 if is_serious else 2
    return 2 if is_serious else 0


def _build_differential(selected_symptoms, patient_state, level):
    """Return the top 3 diagnoses most likely driving the recommendation."""
    differentials = []
    seen_dx = set()

    for sym_id in selected_symptoms:
        for dx in SYMPTOM_DIFFERENTIALS.get(sym_id, []):
            if dx["diagnosis"] not in seen_dx:
                entry = dict(dx)
                entry["source_symptom"] = sym_id

                age = patient_state.age or 40
                sex = patient_state.sex or "unknown"
                pmh = set(patient_state.pmh) if patient_state.pmh else set()

                if "older adults" in dx["notes"].lower() and age < 40:
                    entry["likelihood"] = _demote(dx["likelihood"])
                if "younger patients" in dx["notes"].lower() and age >= 60:
                    entry["likelihood"] = _demote(dx["likelihood"])
                if "reproductive-age female" in dx["notes"].lower() and sex == "male":
                    continue
                if "older males" in dx["notes"].lower() and sex == "female":
                    continue
                if "diabetes" in dx["notes"].lower() and "Diabetes" in pmh:
                    entry["likelihood"] = _promote(dx["likelihood"])
                if "cardiac" in dx["notes"].lower() and "Heart Problems" in pmh:
                    entry["likelihood"] = _promote(dx["likelihood"])
                if "osteoporosis" in dx["notes"].lower() and age >= 65:
                    entry["likelihood"] = _promote(dx["likelihood"])

                differentials.append(entry)
                seen_dx.add(dx["diagnosis"])

    differentials.sort(key=lambda d: (
        _acuity_score(d, level),
        LIKELIHOOD_ORDER.get(d["likelihood"], 3),
    ))
    return differentials[:3]


def _promote(likelihood):
    order = ["Rare", "Uncommon", "Less common", "Common", "Very common"]
    idx = order.index(likelihood) if likelihood in order else 2
    return order[min(idx + 1, len(order) - 1)]


def _demote(likelihood):
    order = ["Rare", "Uncommon", "Less common", "Common", "Very common"]
    idx = order.index(likelihood) if likelihood in order else 2
    return order[max(idx - 1, 0)]
