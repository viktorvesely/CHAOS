class ImageSlide {
    constructor(width, height, x0, y0, name) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;
        this.name = name;
        this.img = null;
        this.recover = backgroundColor;
    }


    start() {
        let img = new Image();

        let ctx = this;

        img.onload = function(){
            let h = img.height;
            let w = img.width;
            let ratio = Math.min(ctx.width / w, ctx.height / h);
            h =  h * ratio;
            w = w * ratio;
            let xo = (ctx.width - w) / 2;
            let yo = (ctx.height - h) / 2;
            img.style.width = `${w}px`;
            img.style.height = `${h}px`;
            img.style.position = 'absolute';
            img.style.top = `${yo}px`;
            img.style.left = `${xo}px`;
            img.style.zIndex = "3";
            document.body.appendChild(img);
        }

        img.src = this.name;
        this.img = img;

        this.recover = backgroundColor;
        backgroundColor = "#ffffff"; 
        el_body.style.backgroundColor = "#ffffff";
    }

    draw() {

    }

    end() {
        this.img.remove();
        backgroundColor = this.recover;
        el_body.style.backgroundColor = backgroundColor;
    }
}