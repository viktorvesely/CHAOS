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
    },
    pause: () => {
        paused = !paused;
    },
    step: () => {
        frameId++;
    },
    showK_o: false,
    showRho: false,
    showInjectors: false,
    showDetectors: false,
    speed: 1,
    trail: 30,
    trajectory: false,
    disturbed: false
}

gui.add(options, "reset");
gui.add(options, "pause");
gui.add(options, "step");
gui.add(options, "speed").min(0).max(2).step(0.01);
gui.add(options, "showK_o");
gui.add(options, "showRho");
gui.add(options, "showInjectors");
gui.add(options, "showDetectors");
gui.add(options, "trail").min(1).max(100).step(1);
gui.add(options, "trajectory");
gui.add(options, "disturbed");

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
    t = frameId * dt;
    d = Math.floor(t / fs);
    text.innerText = `d: ${d} \n t: ${t.toFixed(5)}`;
}

function init() {
    frameId = 0;

}

function trad(i) {
    i = i % trajectory_d[0].length
    
    let x = trajectory_d[0][i];
    let y = trajectory_d[1][i];

    return [x * width, y * height];
}

function tra(i) {
    i = i % trajectory[0].length
    
    let x = trajectory[0][i];
    let y = trajectory[1][i];

    return [x * width, y * height];
}

function draw_trajectory() {
    let ri = Math.round(frameId);
    let src;

    src = tra(ri);
    ctx.beginPath();
    ctx.moveTo(src[0], src[1]);
    for (let i = 1; i < options.trail; i++) {
        src = tra(ri - i);
        ctx.lineTo(src[0], src[1]);
    }
    ctx.strokeStyle = "white";
    ctx.stroke();


    if (options.disturbed) {
        src = trad(ri);
        ctx.beginPath();
        ctx.moveTo(src[0], src[1]);
        for (let i = 1; i < options.trail; i++) {
            src = trad(ri - i);
            ctx.lineTo(src[0], src[1]);
        }
        ctx.strokeStyle = "red";
        ctx.stroke();
    }
}

function drawgrid() {

    let index = Math.round(frameId);

    frame = data[index];


    for (let y = 0; y < gridy; y++) {
        for (let x = 0; x < gridx; x++) {
            let v = frame[y][x] * 255;
            let g = g_k[y][x] * 255;
            let r = rho[y][x] * 255;

            if (options.showK_o) {
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

    if (options.showInjectors) {
        injectors.forEach(injector => {
            let x, y;
            
            x = injector[1];
            y = injector[0];

            ctx.beginPath();
            ctx.fillStyle ="rgb(30, 232, 229)";   
            ctx.rect(x * tileW, y * tileH, tileW, tileH);
            ctx.fill();
            
        });
    }

    if (options.showDetectors) {
        detectors.forEach(detector => {
            let x, y;
            
            x = detector[1];
            y = detector[0];

            ctx.beginPath();
            ctx.fillStyle ="rgb(245, 126, 66)";   
            ctx.rect(x * tileW, y * tileH, tileW, tileH);
            ctx.fill();
            
        });
    }

} 

function draw() {

    ctx.fillStyle = "black"; 
    ctx.beginPath();
    ctx.rect(0, 0, width, height);    
    ctx.fill();

    let index = Math.round(frameId);

    if (index >= data.length) {
        frameId = 0;
        index = 0;
    }
    

    if (options.trajectory) {
        draw_trajectory();
    } else {
        drawgrid();
    }

    time();
    
    if (!paused) frameId += options.speed;

    requestAnimationFrame(draw);
}


window.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft") {
        frameId--;
    } else if (e.key === "ArrowRight") {
        frameId++;
    }
});

rescale();
init();
requestAnimationFrame(draw);