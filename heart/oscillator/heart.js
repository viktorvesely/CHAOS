class Heart {
    constructor(width, height, x0, y0, video, rho) {
        this.width = width;
        this.height = height;
        this.titleSize = 100;
        this.hh = height - this.titleSize;
        this.hy0 = y0 + this.titleSize;
        this.x0 = x0;
        this.y0 = y0;

        this.rho = rho;

        this.video = video;

        this.frameId = 0;
        this.speed = 0.5;
        
        this.rows = this.video[0].length;
        this.cols = this.video[0][0].length;

        this.tileSize = Math.min(this.hh / this.rows, this.width / this.cols);

        this.xOffset = (this.width - this.tileSize * this.cols) / 2;
        this.yOffset = (this.hh - this.tileSize * this.rows) / 2;

        this.rhoMask = false;
        this.title = "Healthy full heart";


    }

    fast() {
        this.speed = 0.5;
    }

    slow() {
        this.speed = 0.1;
    }

    renderTitle(ctx) {
        ctx.fillStyle = primaryColor;
        ctx.textAlign = "center";
        ctx.font = titleFont;
        ctx.fillText(this.title, this.x0 + this.width / 2, this.y0 + 50);
    }

    drawInjectors(ctx) {
        const size = this.tileSize;
        const xo = this.xOffset;
        const yo = this.yOffset;  
        injectors.forEach((injector, i) => {
            let x = injector[1];
            let y = injector[0];

            ctx.beginPath();
            ctx.fillStyle = a_cols[i];
            ctx.rect(this.x0 + xo + x * size, this.y0 + yo + y * size + this.titleSize, size, size);
            ctx.fill();
        });
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

                if (this.rhoMask) {
                    let g = this.rho[y][x] * 255;
                    ctx.fillStyle =`rgb(${v}, ${g}, ${v})`;
                } else {
                    ctx.fillStyle =`rgb(${v}, 0, ${v})`;
                    
                    //ctx.fillStyle =`rgb(255, ${255 - v}, 255)`;   
                }

                ctx.beginPath();
                ctx.rect(this.x0 + xo + x * size, this.y0 + yo + y * size + this.titleSize, size, size);
                ctx.fill();
            }
        }
        
        this.frameId += this.speed;

        ctx.beginPath();
        ctx.strokeStyle = primaryColor;
        ctx.rect(this.x0 + xo, this.y0 + yo + this.titleSize, this.width - xo * 2, this.hh);
        ctx.stroke();
        this.renderTitle(ctx);
    }


    start() { }


    end() { }
}