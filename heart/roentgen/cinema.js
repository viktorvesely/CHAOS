var canvas = document.getElementById("movie");
var text = document.getElementById("output");
var ctx = canvas.getContext("2d");
var paused = false;
var frameId = null;

var gridy = data[0].length;
var gridx = data[0][0].length;

var width = null;
var height = null;
var tileW = null;
var tileH = null;

var gui = new dat.GUI();
var options = {
    reset: () => {
        init();
        console.log(frameId);
    },
    pause: () => {
        paused = !paused;
    },
    step: () => {
        frameId++;
    },
    showG_k: false,
    showRho: false,
    speed: 1
}

gui.add(options, "reset");
gui.add(options, "pause");
gui.add(options, "step");
gui.add(options, "showG_k");
gui.add(options, "showRho");
gui.add(options, "speed").min(0).max(2).step(0.01);

function rescale() {
    let x, y;
    x = window.innerHeight; //window.innerWidth;
    y = window.innerHeight;

    canvas.width = x;
    canvas.height = y;

    width = x;
    height = y;

    dim = Math.min(x / gridx, y / gridy)
    tileW = dim;
    tileH = dim;
}

function time() {
    text.innerText = (frameId * dt).toFixed(5).toString();
}

function init() {
    frameId = 0;

}


function draw() {
    let index = Math.round(frameId);

    if (index >= data.length) {
        frameId = data.length - 1;
        index = data.length - 1;
    };
    
    frame = data[index];

    ctx.fillStyle = "black"; 
    ctx.beginPath();
    ctx.rect(0, 0, width, height);    
    ctx.fill();

    for (let y = 0; y < gridy; y++) {
        for (let x = 0; x < gridx; x++) {
            let v = frame[y][x] * 255;
            let g = g_k[y][x] * 255;
            let r = rho[y][x] * 255;

            if (options.showG_k) {
                ctx.fillStyle =`rgb(0, 0, ${g})`;   
            } else if (options.showRho) {
                ctx.fillStyle =`rgb(${r}, 0, 0)`;   
            } else {
                ctx.fillStyle =`rgb(${v}, 0, ${v})`;   
            }

            ctx.beginPath();
            ctx.rect(x * tileW, y * tileH, tileW, tileH);
            ctx.fill();
        }
    }

    time();
    
    if (!paused) frameId += options.speed;
    requestAnimationFrame(draw);
}

rescale();
init();
requestAnimationFrame(draw);