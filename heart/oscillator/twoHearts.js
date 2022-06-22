class TwoHearts {

    constructor(width, height, x0, y0, h_video, c_video, rho) {
        this.width = width;
        this.height = height;
        this.x0 = x0;
        this.y0 = y0;

        this.hHeart = new Heart(width / 2, height, x0, y0, h_video, null);
        this.cHeart = new Heart(width / 2, height, x0 + width / 2, y0, c_video, rho);
        
        this.cHeart.title = "Scarred heart";
        this.hHeart.title = "Healthy heart";
    }

    start() {

    }

    end() {

    }

    draw(ctx) {
        this.cHeart.draw(ctx);
        this.hHeart.draw(ctx);
    }

    showMasks() {
        this.cHeart.rhoMask = true;
    }
    
}