
var canvas = document.getElementById("screen");
var ctx = canvas.getContext("2d");
var width = -1, height = -1;
const trailSize = 100;
var s0 = [0.5, 0.0];
var state = s0;
var hState = [0.50000001, 0.00000001];
var trail = [state];
var hTrail = [hState];

const omega = 3.37015;
const dt = 0.02;
const a = 5, b = 5;
const min = -8, max = 8;
const range = (max - min);
var rescaleX = -1;
var rescaleY = -1;

function resize() {
    let s = Math.min(window.innerHeight, window.innerWidth) - 5;
    width = height = s;
    canvas.width = width;
    canvas.height = height;
    rescaleX = width / range;
    rescaleY = height / range;
}

resize();

function add(vec1, vec2) {
    return [
        vec1[0] + vec2[0],
        vec1[1] + vec2[1]
    ]
}

function scale(vec, scalar) {
    return [
        vec[0] * scalar,
        vec[1] * scalar
    ]
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

function rescale(state) {
    return [
        (state[0] - min) * rescaleX,
        (state[1] - min) * rescaleY
    ]
}

var t = 0.0;
var actionStrength = 1.0;
var nCycles = 5;
var keys = {
    "a": false, "d": false, "w": false, "s": false
};


function drawTrail(trail, healthy) {
    let from, to;

    from = rescale(trail[0]);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    for (let i = 1; i < trail.length; i++) {
        to = rescale(trail[i]);
        ctx.lineTo(to[0], to[1]);
    }
    ctx.strokeStyle = healthy ? "#fc03ec" : "#0000ff";
    ctx.stroke();
}

function draw() {

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width, height);

    let action = 0.0;
    if (keys["w"])  {
        actionStrength += 0.01;
    }
    if (keys["s"])  {
        actionStrength -= 0.01;
    }
    if (keys["a"])  {
        action -= actionStrength;
        console.log(action);
    }
    if (keys["d"])  {
        action += actionStrength;
        console.log(action);
    }

    // let delta = dsdt(state, t, a, b, omega, action);
    // let hDelta = dsdt(hState, t, a, b, omega, 0.0);
    // t += dt;
    // state = update(state, delta, dt);
    // hState = update(hState, hDelta, dt);

    for (let i = 0; i < nCycles; i++) {
        let delta = dsdt(state, t, a, b, omega, action);
        let hDelta = dsdt(hState, t, a, b, omega, action);
        t += dt;
        state = update(state, delta, dt);
        hState = update(hState, hDelta, dt);
    }

    trail.push(state);
    hTrail.push(hState);

    if (trail.length > trailSize) {
        trail.shift();
    }

    if (hTrail.length > trailSize) {
        hTrail.shift();
    }

    drawTrail(trail, false);
    drawTrail(hTrail, true);

    requestAnimationFrame(draw);
}

window.addEventListener("keydown", e => {
    keys[e.key] = true;
});

window.addEventListener("keyup", e => {
    keys[e.key] = false;
});

requestAnimationFrame(draw);