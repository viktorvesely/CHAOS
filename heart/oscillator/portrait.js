const trailSize = 100;
var nCycles = 1;

class Portrait {
    constructor(width, height, x0, y0) {
        this.w = width;
        this.h = height;


        this.xbounds = [-6, 6];
        this.ybounds = [-9, 9];
        this.x0 = x0;
        this.y0 = y0;

        this.xColor = "#0000ff";
        this.yColor = "#00ff00";

        this.title = "Phase portrait"
        this.lineWidth = 2.5;
    }

    stateToCtx(state) {
        
        let x = state[0];
        let y = state[1];
        let xbound = this.xbounds;
        let ybound = this.ybounds;
        
        let pos01 = [
            (x - xbound[0]) / (xbound[1] - xbound[0]),
            (y - ybound[0]) / (ybound[1] - ybound[0])
        ]

        if (pos01[0] < 0 || pos01[0] > 1 || pos01[1] < 0 || pos01[1] > 1) {
            return false;
        }

        return [
            pos01[0] * this.w + this.x0,
            pos01[1] * this.h + this.y0
        ]
    }

    pos01ToCtx(pos) {
        return [
            pos[0] * this.w + this.x0,
            pos[1] * this.h + this.y0
        ]
    }

    drawTrail(ctx, trail, color) {
        let from, to;
    
        from = this.stateToCtx(trail[0]);
        ctx.lineWidth = this.lineWidth;
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        for (let i = 1; i < trail.length; i++) {
            to = this.stateToCtx(trail[i]);
            if (to === false) {
                break;
            }
            ctx.lineTo(to[0], to[1]);
        }
        ctx.strokeStyle = color;
        ctx.stroke();
    }

    renderTitle(ctx) {
        let pos = this.pos01ToCtx([0.5, 0.1]);

        ctx.fillStyle = "#ffffff";
        ctx.font = "22px Verdana";
        ctx.fillText(this.title, pos[0] - 50, pos[1]);
    }

    axis(ctx, lb01, lt01, rb01, xlab, ylab) {
        let lb = this.pos01ToCtx(lb01);
        let lt = this.pos01ToCtx(lt01);
        let rb = this.pos01ToCtx(rb01);

        ctx.beginPath();
        ctx.strokeStyle = this.xColor;
        ctx.moveTo(lb[0], lb[1]);
        ctx.lineTo(rb[0], rb[1]);
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = this.yColor;
        ctx.moveTo(lb[0], lb[1]);
        ctx.lineTo(lt[0], lt[1]);
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "15px Verdana";
        ctx.fillText(xlab, lb[0] + (rb[0] - lb[0]) / 2, rb[1] + 19);
        ctx.fillText(ylab, lb[0] - 50, lt[1] + (lb[1] - lt[1]) / 2);
    }
    
}

