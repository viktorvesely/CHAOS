const graphTrail = 400;

class Graph {
    constructor(x0, y0, width, height, twindow, vbound, color, axisColor) {
        this.x0 = x0;
        this.y0 = y0;
        this.width = width;
        this.height = height;
        this.twindow = twindow;
        this.vbound = vbound;
        this.color = color;
        this.axisColor = axisColor;

        this.t = 0;
        this.tStack = [];

        this.title = "";
    }

    pos01toCtx(pos) {
        return [
            pos[0] * this.width + this.x0,
            pos[1] * this.height + this.y0
        ]
    }

    pos01ToVal(pos) {
        let tbound = [
            this.t + this.twindow[0],
            this.t + this.twindow[1]
        ]

        let vbound = this.vbound;
        
        return [
            pos[0] * (tbound[1] - tbound[0]) + tbound[0],
            pos[1] * (vbound[1] - vbound[0]) + vbound[0]
        ]
    }

    pushTime() {
        this.tStack.push(this.t);
    }

    popTime() {
        this.t = this.tStack.pop();
    }

    valToCtx(t, val) {
        
        let tbound = [
            this.t + this.twindow[0],
            this.t + this.twindow[1]
        ]

        let vbound = this.vbound;
        
        let pos01 = [
            (t - tbound[0]) / (tbound[1] - tbound[0]),
            (val - vbound[0]) / (vbound[1] - vbound[0])
        ]

        if (pos01[0] < 0 || pos01[0] > 1 || pos01[1] < 0 || pos01[1] > 1) {
            return false;
        }

        return [
            pos01[0] * this.width + this.x0,
            pos01[1] * this.height + this.y0
        ]
    }

    drawTitle(ctx) {
        let pos = this.pos01toCtx([0.45, 0.1]);
        ctx.font = "22px Verdana";
        ctx.fillStyle = this.axisColor;
        ctx.fillText(this.title, pos[0], pos[1]);
    }

    axis(ctx) {
        let fromA = this.pos01toCtx([0, 0.9]);
        let toA = this.pos01toCtx([1, 0.9]);
        let valaxis = this.pos01ToVal([0, 0.9])[1];
        
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(fromA[0], fromA[1]);
        ctx.lineTo(toA[0], toA[1]);
        ctx.strokeStyle = this.axisColor;
        ctx.stroke();

        let t_min = this.t + this.twindow[0];
        let t_max = this.t + this.twindow[1];

        const step = 1;
        let tick_max = Math.ceil(t_max + 1);
        let tick_min = Math.floor(t_min - 1);

        let ticks = Math.floor((tick_max - tick_min) / step) - 1;
        const tickl = 10;

        for (let i = 0; i < ticks; i++) {
            let x = tick_max - i * step;
            let from = this.valToCtx(x, valaxis);
            let to = this.valToCtx(x, valaxis);

            if (from === false || to === false) {
                continue;
            }

            from[1] -= tickl;
            to[1] += tickl;
            
            ctx.beginPath();
            ctx.moveTo(from[0], from[1]);
            ctx.lineTo(to[0], to[1]);
            ctx.stroke();

        }

        ctx.font = "18px Verdana";
        ctx.fillStyle = this.color;
        ctx.fillText("Time", toA[0] - 100, toA[1] + 25);
    }

    legend(ctx, xCol, yCol, xtext="AP", ytext="AP/dt", xoffset=120, yoffset=120) {
    
        ctx.lineWidth = 2.5;
        if (xtext.length > 0) {
            let row1 = this.pos01toCtx([1.0, 0.22]);
    
            ctx.font = "14px Verdana";
            ctx.fillStyle = this.axisColor;
            ctx.fillText(xtext, row1[0] - xoffset, row1[1] + 5);
            ctx.beginPath();
            ctx.moveTo(row1[0] - 80, row1[1]);
            ctx.lineTo(row1[0] - 20, row1[1]);
            ctx.strokeStyle = xCol;
            ctx.stroke();
        }

        if (ytext.length > 0) {
            let row2 = this.pos01toCtx([1.0, 0.26]);

            ctx.fillText(ytext, row2[0] - yoffset, row2[1] + 5);
            ctx.beginPath();
            ctx.moveTo(row2[0] - 80, row2[1]);
            ctx.lineTo(row2[0] - 20, row2[1]);
            ctx.strokeStyle = yCol;
            ctx.stroke();
        }
    }

    draw(ctx, trail, index, dt, scrolling) {
        let t = this.t;
        let val = trail[trail.length - 1][index];
        let from = this.valToCtx(t, val);
        let to;
        t -= dt;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        for (let i = trail.length - 2; i >= 0; --i) {

            val = trail[i][index];
            to = this.valToCtx(t, val);
            if (to === false) {
                break;
            }
            ctx.lineTo(to[0], to[1]);
            from = to;
            t -= dt;
        }
        ctx.strokeStyle = this.color;
        ctx.stroke();

        this.t += scrolling;
    }
}


class CompGraphs {
    constructor(width, height, x0, y0) {
        this.w = width;
        this.h = height;

        this.state = [0.5 , 0.0];
        this.hstate = [0.5, 0.0];
        
        this.trail  = [this.state];
        this.htrail = [this.hstate];
        this.t = 0;

        this.x0 = x0;
        this.y0 = y0;

        this.xColor = "#0000ff";
        this.yColor = "#00ff00";
        this.axisColor = "#ffffff";

        this.hgraph = new Graph(
            x0, y0,
            width / 2, height,
            [-8, 4],
            [-8.5, 8.5],
            this.xColor,
            this.axisColor
        )

        this.graph = new Graph(
            x0 + width / 2, y0,
            width / 2, height,
            [-8, 4],
            [-8.5, 8.5],
            this.xColor,
            this.axisColor,
        );

        this.graph.title = "Arrhythmia";
        this.hgraph.title = "Healthy";
    }

    draw(ctx) {
        let delta = dsdt(this.state, this.t, a, b, omega, 0);
        let hdelta = dsdt(this.hstate, this.t, a, b, 1.0, 0);
        this.t += dt;
        this.state = update(this.state, delta, dt);
        this.hstate = update(this.hstate, hdelta, dt);

        this.trail.push(this.state);
        this.htrail.push(this.hstate);
        
        if (this.htrail.length > graphTrail) {
            this.htrail.shift();
        }

        if (this.trail.length > graphTrail) {
            this.trail.shift();
        }
        
        this.graph.color = this.yColor;
        this.graph.pushTime();
        this.graph.draw(ctx, this.trail, 1, dt, dt);
        this.graph.popTime();
        this.graph.color = this.xColor;
        this.graph.draw(ctx, this.trail, 0, dt, dt);
        this.graph.color = this.axisColor;
        this.graph.axis(ctx);
        this.graph.drawTitle(ctx);
        this.graph.legend(ctx, this.xColor, this.yColor, "AP", "AP/dt", 110, 130);

        this.hgraph.color = this.yColor;
        this.hgraph.pushTime();
        this.hgraph.draw(ctx, this.htrail, 1, dt, dt);
        this.hgraph.popTime();
        this.hgraph.color = this.xColor;
        this.hgraph.draw(ctx, this.htrail, 0, dt, dt);
        this.hgraph.color = this.axisColor;
        this.hgraph.axis(ctx);
        this.hgraph.drawTitle(ctx);
        this.hgraph.legend(ctx, this.xColor, this.yColor, "AP", "AP/dt", 110, 130);

        ctx.beginPath();
        ctx.moveTo(this.w / 2, this.y0);
        ctx.lineTo(this.w / 2, this.y0 + this.h);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    start() {

    }
}

class IntroGraph {
    constructor(width, height, x0, y0) {
        this.w = width;
        this.h = height;

        this.state = [0.5 , 0.0];
        
        this.xtrail  = [[this.state[0]]];
        this.ytrail = [];
        this.t = 0;

        this.x0 = x0;
        this.y0 = y0;

        this.xColor = "#0000ff";
        this.yColor = "#00ff00";
        this.axisColor = "#ffffff";

        this.trailSize = 500;


        this.graph = new Graph(
            x0, y0,
            width, height,
            [-10, 4],
            [-8.5, 8.5],
            this.xColor,
            this.axisColor,
        );

        this.graph.title = "Heartbeat";
    }

    draw(ctx) {
        let delta = dsdt(this.state, this.t, a, b, 1.0, 0);
        this.t += dt;
        this.state = update(this.state, delta, dt);

        this.xtrail.push([this.state[0]]);
        
        if (this.xtrail.length > this.trailSize) {
            this.xtrail.shift();
        }
        
        this.graph.color = this.xColor;
        this.graph.draw(ctx, this.xtrail, 0, dt, dt); 
        this.graph.color = this.axisColor;
        this.graph.axis(ctx);
        this.graph.drawTitle(ctx);
        this.graph.legend(ctx, this.xColor, this.yColor, "Action potential", "", 200);

    }

    start() {

    }
}

class ExtendIntro {
    constructor(width, height, x0, y0, introGraph) {
        this.w = width;
        this.h = height;
        this.introGraph = introGraph;

        this.state = [0.5 , 0.0];
        
        this.xtrail  = [];
        this.ytrail = [];
        this.t = 0;

        this.x0 = x0;
        this.y0 = y0;

        this.xColor = "#0000ff";
        this.yColor = "#00ff00";
        this.axisColor = "#ffffff";

        this.trailSize = 500;

        this.graph = new Graph(
            x0, y0,
            width, height,
            [-10, 4],
            [-8.5, 8.5],
            this.xColor,
            this.axisColor,
        );

        this.graph.title = "Heartbeat";
    }

    draw(ctx) {
        let delta = dsdt(this.state, this.t, a, b, 1.0, 0);
        this.t += dt;
        this.state = update(this.state, delta, dt);

        this.xtrail.push([this.state[0]]);
        this.ytrail.push([this.state[1]])
        
        if (this.xtrail.length > this.trailSize) {
            this.xtrail.shift();
        }

        if (this.ytrail.length > this.trailSize) {
            this.ytrail.shift();
        }
        
        this.graph.color = this.xColor;
        this.graph.pushTime();
        this.graph.draw(ctx, this.xtrail, 0, dt, dt); 
        this.graph.popTime();
        this.graph.color = this.yColor;
        this.graph.draw(ctx, this.ytrail, 0, dt, dt); 
        this.graph.color = this.axisColor;
        this.graph.axis(ctx);
        this.graph.drawTitle(ctx);
        this.graph.legend(ctx, this.xColor, this.yColor, "Action potential", "Rate of change", 200, 200);

    }

    start() {
        this.ytrail = [];
        this.state = this.introGraph.state;
        this.xtrail = this.introGraph.xtrail;
        this.t = this.introGraph.t;
    }
}
