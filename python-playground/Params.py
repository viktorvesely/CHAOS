class ModelData:
    HR = 0 # Heart rate Hz
    TL = 0 # Heart beat period s
    TL0 = 0 # Intrinsic heart beat period of the s
    # Sinoatrial node (constant)
    CVN  = 0 # Integrated cardiac vagal nerve signal —
    CVN0 = 0 # Unmodulated central parasympathetic —
    # Vagal activity (constant)
    SM = 0 # Integrated cardiac sympathetic signal (constant) —
    S1 = 0 # Response frequencies of TL and Hz CVN, respectively (constants)
    S2 = 0 # Response frequencies of TL and Hz CVN, respectively (constants)
    c0 = 0 # Parasympathetic input to the heart —
    c1 = 0 # Central respiratory signal Hz
    c2 = 0 # Mechanical feedback from the lungs (s L)1
    c3 = 0 # Baroreflex input to the heart Hz
    k2 = 0 # Constant affecting the mechanical L
    # Feedback from the lungs
    G = 0 # Central-respiratory baroreflex gating function —
    a = 0 # Constant affecting the slope of the gating function s/L
    BR = 0 # Baroreflex input to cardiac vagal tone —
    Rs = 0 # Systemic resistance (constant) mmHg s/L
    Vc = 0 # Stroke volume (constant) L
    d = 0 # Ideal mean arterial pressure (constant) mmHg
    c = 0 # Constant affecting the strength of the baroreflex mmHg1
    k3 = 0 # Constant affecting the baroreceptor signal —
    MAP = 0 # Mean arterial pressure mmHg
    A = 0 # Central respiratory rhythm-generating signal —
    Rp = 0 # Phrenic nerve signal —
    q = 0 # Airflow into the lungs L/s
    VA = 0 # Lung volume 