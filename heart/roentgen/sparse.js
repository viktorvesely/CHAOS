
var canvas = document.getElementById("sketch");
var ctx = canvas.getContext("2d");
var n = array.length;

var offsetY = 60;
var offsetX = 60;
var spacing = 70;
var angleArrow = 10;
var arrowLength = 0;

var width = window.innerWidth;
var height = window.innerHeight - 10;

canvas.width = width;
canvas.height = height;

var perRow = Math.round(Math.sqrt(n));
var rows = Math.ceil(n / perRow);

var sizeX = width / perRow;
var sizeY = height / rows;

var size = Math.min(10, Math.min(sizeX, sizeY));

var nodes = [];

var alreadyDrew = 0;
outer:
for (let y = 0; y < rows; y++) {
    for (let x = 0; x < perRow; x++) {
        
        let cx = offsetX + x * (size * 2 + spacing);
        let cy = offsetY + y * (size * 2 + spacing);
        nodes.push([cx, cy, "#886694"]);

        ++alreadyDrew;
        if (alreadyDrew >= n) {
            break outer;
        }
    }
}


function draw() {

    ctx.fillStyle = "black"; 
    ctx.beginPath();
    ctx.rect(0, 0, width, height);    
    ctx.fill();

    for (let to = 0; to < n; to++) {
        for (let from = 0; from < n; from++) {

            if (array[to][from] === 0) {
                continue;
            } 

            if (to === from) {
                nodes[to][2] = "#e612c2";
                continue;
            }
            let srcNode, s, destNode, d;

            s = srcNode = nodes[from];
            d = destNode = nodes[to];

            let style = null;

            if (array[from][to] !== 0) {
                style = "white";
            }
            else {
                style = ctx.createLinearGradient(s[0], s[1], d[0], d[1]);
                style.addColorStop(0.1, "red");
                style.addColorStop(1, "white");
            }

            ctx.strokeStyle = style;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(srcNode[0], srcNode[1]);
            ctx.lineTo(destNode[0], destNode[1]);
            ctx.stroke();
        }
    }

         
    nodes.forEach(node => {
        let cx = node[0];
        let cy = node[1];
        ctx.fillStyle = node[2];
        ctx.beginPath();
        ctx.arc(cx, cy, size, 0, 2 * Math.PI);
        ctx.fill();
    });

}


requestAnimationFrame(draw);

