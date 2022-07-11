
var canvas = document.getElementById("screen");
var ctx = canvas.getContext("2d");
var width = -1, height = -1;


const titleFont = "34px Verdana";
const otherFont = "24px Verdana";
const legendFont = "18px Verdana";

// const primaryColor = "#ffffff";
// var backgroundColor = "#000000";
const primaryColor = "#000000";
var backgroundColor = "#ffffff";
const yColor = "#07db3f";

var el_body = document.getElementsByTagName("body")[0];

el_body.style.backgroundColor = backgroundColor;

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
var heart = new Heart(width, height, 0, 0, h_v_data, rho);
var twoHearts = new TwoHearts(width, height, 0, 0, h_v_data, c_v_data, rho);
var simpleController = new ImageSlide(width, height, 0, 0, "controller_simple.png");
var actor = new Actor(width, height, 0, 0, a_v_data);
var explanation= new ImageSlide(width, height, 0, 0, "con_explanation.png");
var equations = new ImageSlide(width, height, 0, 0, "equations.png");
var con_exploit = new ImageSlide(width, height, 0, 0, "controller_exploit.png");
var arch_heart = new ImageSlide(width, height, 0, 0, "arch_heart.png");
var arch_local = new ImageSlide(width, height, 0, 0, "arch_local.png");
var arch_pca = new ImageSlide(width, height, 0, 0, "arch_pca.png");
var disscussion = new Disscussion(width, height, 0, 0);
var metric = new ImageSlide(width, height, 0, 0, "metric.png");
var results = new ImageSlide(width, height, 0, 0, "results.png");
var val_graph = new ImageSlide(width, height, 0, 0, "val_graph.png");
var test_graph = new ImageSlide(width, height, 0, 0, "test_graph.png");
var dis_text = new ImageSlide(width, height, 0, 0, "dis_text.png");
var title = new ImageSlide(width, height, 0, 0, "title.png");


var frames = [
    title,
    intro,
    extendIntro,
    compareGraph,
    phaseTime,
    chaos,
    new SlideEvent(() => { chaos.faster(); }),
    chaos,
    new SlideEvent(() => { chaos.slower(); }),
    chaos,
    twoHearts,
    new SlideEvent(() => { twoHearts.showMasks(); }),
    twoHearts,
    equations,
    simpleController,
    explanation,
    actor,
    con_exploit,
    arch_heart,
    arch_local,
    arch_pca,
    metric,
    results,
    val_graph,
    test_graph,
    dis_text,
    disscussion
]

var activeFrame = 0;
frames[activeFrame].start();

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

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);


    if (press("d")) {
        frames[activeFrame].end();
        activeFrame++;
        if (activeFrame >= frames.length) {
            activeFrame = 0;
        }
        frames[activeFrame].start(1);
    }

    if (press("a")) {
        frames[activeFrame].end();
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


window.addEventListener("click", e => {
    frames[activeFrame].end();
    activeFrame++;
    if (activeFrame >= frames.length) {
        activeFrame = 0;
    }
    frames[activeFrame].start(1);
})

window.oncontextmenu = function ()
{
    frames[activeFrame].end();
    activeFrame--;
    if (activeFrame < 0) {
        activeFrame = frames.length - 1;
    }

    frames[activeFrame].start(-1);
    return false;     // cancel default menu
}

requestAnimationFrame(draw);