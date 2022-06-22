const a_1_col = "#ff0000";
const a_2_col = "#00ff00";
const a_3_col = "#0000ff";
const a_4_col = "#ffd60a";

const a_cols = [a_1_col, a_2_col, a_3_col, a_4_col];

class Actor {
    constructor(width, height, x0, y0, actor_video) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;

        this.heart= new Heart(width / 2, height, x0, y0, actor_video);
        this.heart.title = "Controlled heart";
        this.graph = new Graph(x0 + width / 2, y0, width / 2, height, [-7, 4], [-0.5, 3], a_1_col, primaryColor);
        this.graph.title = "Actions";

        this.trailSize = 350;
        this.t = 0;

        this.a1 = [[this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()]];
        this.a2 = [[this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()]];
        this.a3 = [[this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()]];
        this.a4 = [[this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()], [this.f(), this.p(), this.a()]];
        this.trail = [this.getAction()];

        this.dt = 0.04;

    }

    f() {
        return Math.random() * 5 - 2;
    }

    p() {
        return Math.random() * Math.PI * 2;
    }

    a() {
        return Math.random() * 1.5;
    }

    cos(args) {
        const t = this.t;
        return Math.cos(args[0] * t + args[1]) * args[2];
    }

    getAction() {
        return [
            Math.min(Math.max(0, this.cos(this.a1[0]) + this.cos(this.a1[1]) + this.cos(this.a1[2])), 3),
            Math.min(Math.max(0, this.cos(this.a2[0]) + this.cos(this.a2[1]) + this.cos(this.a2[2])), 3),
            Math.min(Math.max(0, this.cos(this.a3[0]) + this.cos(this.a3[1]) + this.cos(this.a3[2])), 3),
            Math.min(Math.max(0, this.cos(this.a4[0]) + this.cos(this.a4[1]) + this.cos(this.a4[2])), 3)
        ]
    }

    start() {

    }

    end() {

    }

    draw(ctx) {
        this.heart.draw(ctx);
        this.heart.drawInjectors(ctx);

        this.graph.pushTime();
        this.graph.color = a_1_col;
        this.graph.draw(ctx, this.trail, 0, dt, dt);
        this.graph.popTime();

        this.graph.pushTime();
        this.graph.color = a_2_col;
        this.graph.draw(ctx, this.trail, 1, dt, dt);
        this.graph.popTime();

        this.graph.pushTime();
        this.graph.color = a_4_col;
        this.graph.draw(ctx, this.trail, 3, dt, dt);
        this.graph.popTime();

        this.graph.color = a_3_col;
        this.graph.draw(ctx, this.trail, 2, dt, dt);

        this.graph.axis(ctx);
        this.graph.drawTitle(ctx);

        this.graph.make_legend(ctx, a_cols, ["a1", "a2", "a3", "a4"], [120, 120, 120, 120]);

        this.t += dt;
        
        this.trail.push(this.getAction());
        
        if (this.trail.length > this.trailSize) {
            this.trail.shift();
        }

        ctx.beginPath();
        ctx.moveTo(this.x0 + this.width / 2, this.y0);
        ctx.lineTo(this.x0 + this.width / 2, this.y0 + this.height);
        ctx.strokeStyle = primaryColor;
        ctx.stroke();
    }
}