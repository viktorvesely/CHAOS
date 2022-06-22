

class PhaseTime {

    constructor(width, height, x0, y0) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;

        this.xcol = "#0000ff";
        this.ycol = yColor;

        this.state = [0.5, 0.0];
        this.trail = [this.state];
        this.phase = new Portrait(width, height / 2, x0, y0 + height / 2);
        this.time = new Graph(
            x0, y0,
            width, height / 2,
            [-10, 4],
            [-8.5, 8.5],
            this.xcol,
            "#ffffff"
        );

        this.time.title = "Time domain"

        this.t = 0;
        this.trailSize = 500;

    }

    start() {

    }

    end() {

    }


    draw(ctx) {
        
        let delta = dsdt(this.state, this.t, a, b, omega, 0);
        this.state = update(this.state, delta, dt);
        this.trail.push(this.state);
        this.t += dt;
        if (this.trail.length > this.trailSize) {
            this.trail.shift();
        }
 
        this.phase.drawTrail(ctx, this.trail.slice(-101, -1),  primaryColor);
        this.phase.axis(ctx, [0.2, 0.85], [0.2, 0.2], [0.8, 0.85], "AP", "AP/dt");
        this.phase.renderTitle(ctx);

        ctx.beginPath();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle =  primaryColor;
        ctx.moveTo(this.x0, this.y0 + this.height / 2);
        ctx.lineTo(this.x0 + this.width, this.y0 + this.height / 2);
        ctx.stroke();

        this.time.pushTime();
        this.time.color = this.ycol;
        this.time.draw(ctx, this.trail, 1, dt, dt);
        this.time.popTime();
        this.time.color = this.xcol;
        this.time.draw(ctx, this.trail, 0, dt, dt);
        this.time.axisColor = primaryColor;
        this.time.color = primaryColor
        this.time.axis(ctx);
        this.time.drawTitle(ctx);
        
        this.time.legend(ctx, this.xcol, this.ycol, "AP", "AP/dt", 115, 140);

    }


}