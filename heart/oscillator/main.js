
var canvas = document.getElementById("screen");
var ctx = canvas.getContext("2d");
var width = -1, height = -1;

const omega = 3.37015;
const dt = 0.02;
const a = 5, b = 5;

function updateMultiple(states, trails, trailSize, t, dt, nCycles, omega) {
    for (let s = 0; s < states.length; ++s) {
        let state = states[s];
        let local_t = t;
        for (let i = 0; i < nCycles; i++) {
            let delta = dsdt(state, local_t, a, b, omega, 0);
            local_t += dt;
            state = update(state, delta, dt);
        }

        states[s] = state;
        trails[s].push(state);

        if (trails[s].length > trailSize) {
            trails[s].shift();
        }
    }

    return t + dt * nCycles;

}

function update(state, delta, dt) {
    return [
        state[0] + delta[0] * dt,
        state[1] + delta[1] * dt
    ]
}

function dsdt(state, t, a, b, omega, action) {
    let x = state[0];
    let y = state[1];
    let delta = [
        y - a * ((x * x * x) / 3 - x),
        - x + b * Math.cos(t * omega) + action
    ]
    return delta

}

function resize() {
    height = window.innerHeight - 5;
    width = window.innerWidth - 5;
    canvas.width = width;
    canvas.height = height;
}

resize();

var keys = {
    "a": false, "d": false, "w": false, "s": false
};

var depressedKeys = {};

var intro = new IntroGraph(width, height, 0, 0);
var extendIntro = new ExtendIntro(width, height, 0, 0, intro);
var compareGraph = new CompGraphs(width, height, 0, 0);
var phaseTime = new PhaseTime(width, height, 0, 0);
var chaos = new Chaos(width, height, 0, 0);

var frames = [
    intro,
    extendIntro,
    compareGraph,
    phaseTime,
    chaos,
    new SlideEvent(() => { chaos.faster(); }),
    chaos,
    new SlideEvent(() => { chaos.slower(); }),
    chaos
]

var activeFrame = 0;

function updateDepress() {
    for (const [key, value] of Object.entries(keys)) {
        if (value === false) {
            depressedKeys[key] = true;
        }
        else {
            depressedKeys[key] = false;
        }
    }
}

function press(key) {
    return keys[key] && depressedKeys[key];
}

function draw() {

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width, height);


    if (press("d")) {
        activeFrame++;
        if (activeFrame >= frames.length) {
            activeFrame = 0;
        }
        frames[activeFrame].start(1);
    }

    if (press("a")) {
        activeFrame--;
        if (activeFrame < 0) {
            activeFrame = frames.length - 1;
        }

        frames[activeFrame].start(-1);
    }

    frames[activeFrame].draw(ctx);

    updateDepress();
    requestAnimationFrame(draw);
}

window.addEventListener("keydown", e => {
    keys[e.key] = true;
});

window.addEventListener("keyup", e => {
    keys[e.key] = false;
});

requestAnimationFrame(draw);