

function randCol() {
    const min = 0;
    const max = 360;
    return hslToHex(Math.floor(Math.random() * (max - min) + min), 100, 50);
}

function hslToHex(h, s, l) {
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
      const k = (n + h / 30) % 12;
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color).toString(16).padStart(2, '0');   // convert to Hex and prefix "0" if needed
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  }

class Chaos {

    constructor(width, height, x0, y0) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;

        this.xcol = "#0000ff";
        this.ycol = "#00ff00";
    
        let baseState = [0.5, 0.0];
        const nStates = 30;

        this.states = [baseState];
        this.cols = [randCol()];

        for (let i = 0; i < nStates - 1; i++) {
            let phi = Math.random() * Math.PI * 2;
            let r = Math.random() * 0.05;
            
            this.states.push([
                baseState[0] + Math.cos(phi) * r,
                baseState[1] + Math.sin(phi) * r
            ])

            this.cols.push(randCol());
        }

        // this.cols = [
        //     "#07c9fa",
        //     "#ffee00",
        //     "#ff00d9",
        //     "#ffffff"
        // ]

        // this.states = [
        //     [0.5, 0.0],
        //     [0.55, 0.05],
        //     [0.45, 0.0],
        //     [0.45, -0.05]
        // ];

        this.trails = [];

        this.states.forEach(state => {
            this.trails.push([state]);
        });

        this.phase = new Portrait(width, height, x0, y0);
        this.phase.lineWidth = 2;

        this.t = 0;
        this.trailSize = 150;

        this.phase.title = "Chaos showcase";

        this.nCycles = 2;
        this.dt = 0.012;

    }

    faster() {
        this.nCycles = 12;
        this.dt = 0.02;
        this.trailSize = 15;

        let diff = this.trails[0].length - this.trailSize;
        if (diff > 0) {
            this.trails.forEach(trail => {
                trail.splice(0, diff);
            })
        }
    }

    slower() {
        this.nCycles = 2;
        this.dt = 0.012;
        this.trailSize = 150;
    }


    start() {

    }


    draw(ctx) {
        
        this.t = updateMultiple(this.states, this.trails, this.trailSize, this.t, 0.012, this.nCycles, omega);

        this.trails.forEach((trail, i) => {
            this.phase.drawTrail(ctx, trail, this.cols[i]);
        });
        
        this.phase.axis(ctx, [0.2, 0.85], [0.2, 0.2], [0.8, 0.85], "AP", "AP/dt");
        this.phase.renderTitle(ctx);

    }


}