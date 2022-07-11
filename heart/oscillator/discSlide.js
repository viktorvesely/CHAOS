class Disscussion {
    constructor(width, height, x0, y0) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;
        
        this.portrait = new Portrait(width, height, x0, y0);

        this.portrait.title = "Testing NRMSE: 0.1";
        
        this.t = 0;
        this.speed = 0.9;
        this.trailSize = 40;

        this.reference_color = "#ff6f00";
        this.real_color = "#0000ff";
    }

    start() {
        this.t = 0;
    }

    end() {

    }

    draw(ctx) {

        let index = Math.floor(this.t);
        if (index >= reference_traj.length) {
            this.t = 0;
            index = 0;
        }

        let endIndex = index - this.trailSize;

        if (endIndex < 0) {
            endIndex = 0;
        }

        let original = this.portrait.lineWidth;
        let lwreference = original * 3;
        this.portrait.lineWidth = lwreference;
        this.portrait.drawTrail(ctx, reference_traj, this.reference_color, index, endIndex);
        this.portrait.lineWidth = original;
        this.portrait.drawTrail(ctx, real_traj, this.real_color, index, endIndex);

        this.portrait.renderTitle(ctx);
        this.portrait.axis(ctx, [0.2, 0.9], [0.2, 0.2], [0.8, 0.9], "AP", "AP/dt");

        this.portrait.disLegend(ctx, lwreference, original, this.reference_color, this.real_color);
        
        this.t += this.speed;
    }
}