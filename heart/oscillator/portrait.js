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
        this.yColor = yColor;

        this.title = "Phase portrait"
        this.lineWidth = 2.5;
    }

    disLegend(ctx, lwref, lwreal, colRef, colReal) {
        ctx.lineWidth = lwref;
        
        let row1 = this.pos01ToCtx([1.0, 0.22]);

        ctx.font = legendFont;
        ctx.fillStyle = primaryColor;
        ctx.fillText("Reference", row1[0] - 180, row1[1] + 5);
        ctx.beginPath();
        ctx.moveTo(row1[0] - 80, row1[1]);
        ctx.lineTo(row1[0] - 20, row1[1]);
        ctx.strokeStyle = colRef;
        ctx.stroke();

        ctx.lineWidth = lwreal;

        let row2 = this.pos01ToCtx([1.0, 0.26]);

        ctx.fillText("Real", row2[0] - 140, row2[1] + 5);
        ctx.beginPath();
        ctx.moveTo(row2[0] - 80, row2[1]);
        ctx.lineTo(row2[0] - 20, row2[1]);
        ctx.strokeStyle = colReal;
        ctx.stroke();
    
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

    drawTrail(ctx, trail, color, startIndex=-1, endIndex=-1) {
        let from, to;

        if (startIndex !== -1) {
            from = this.stateToCtx(trail[startIndex]);
            let length = startIndex - endIndex;
            ctx.lineWidth = this.lineWidth;
            ctx.beginPath();
            ctx.moveTo(from[0], from[1]);

            for (let i = 1; i < length; i++) {
                to = this.stateToCtx(trail[startIndex - i]);
                if (to === false) {
                    break;
                }
                ctx.lineTo(to[0], to[1]);
            }
        } else {

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
        }

        ctx.strokeStyle = color;
        ctx.stroke();
    }

    renderTitle(ctx) {
        let pos = this.pos01ToCtx([0.5, 0.1]);

        ctx.fillStyle = primaryColor;
        ctx.font = titleFont;
        ctx.textAlign = 'center';
        ctx.fillText(this.title, pos[0], pos[1]);
        ctx.textAlign = 'start';
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

        ctx.fillStyle = primaryColor;
        ctx.font = legendFont;
        ctx.fillText(xlab, lb[0] + (rb[0] - lb[0]) / 2, rb[1] + 19);
        ctx.fillText(ylab, lb[0] - 60, lt[1] + (lb[1] - lt[1]) / 2);
    }
    
}

