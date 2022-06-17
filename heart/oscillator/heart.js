class Heart {
    constructor(width, height, x0, y0, h_video, c_video, rho) {
        this.width = width;
        this.height = height;
        this.hh = height - 100;
        this.hy0 = y0 + 100;
        this.x0 = x0;
        this.y0 = y0;

        this.h_video = h_video;
        this.c_video = c_video;
        this.rho = rho;

        this.video = h_video;

        this.frameId = 0;
        this.speed = 0.5;
        
        this.rows = this.video[0].length;
        this.cols = this.video[0][0].length;

        this.tileSize = Math.min(this.hh / this.rows, this.width / this.cols);

        this.xOffset = (this.width - this.tileSize * this.cols) / 2;
        this.yOffset = (this.height - this.tileSize * this.rows) / 2;

        this.rhoMask = false;
    }

    
    healthy() {
        this.video = this.h_video;
    }

    chaotic() {
        this.video = this.c_video;
    }

    fast() {
        this.speed = 0.5;
    }

    slow() {
        this.speed = 0.2;
    }

    draw(ctx) {
        let n = Math.ceil(this.frameId);

        if (n >= this.video.length) {
            n = 0;
            this.frameId = 0;
        }
        
        let frame = this.video[n];
        const size = this.tileSize;
        const xo = this.xOffset;
        const yo = this.yOffset;

        for (let y = 0; y < this.rows; ++y) {
            for (let x = 0; x < this.cols; ++x) {
                let v = frame[y][x] * 255;
                let r = rho[y][x] * 255;

                if (this.rhoMask) {
                    ctx.fillStyle =`rgb(${r}, 0, 0)`;
                } else {
                    ctx.fillStyle =`rgb(${v}, 0, ${v})`;   
                }

                ctx.beginPath();
                ctx.rect(this.x0 + xo + x * size, this.y0 + yo + y * size, size, size);
                ctx.fill();
            }
        }
        
        this.frameId += this.speed;
    }


    start() { }


    end() { }
}