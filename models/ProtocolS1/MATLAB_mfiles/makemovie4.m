clear all; close all;

path = '/Users/roel/Documents/Manuscripts/PLoS-Rene/Code/RFMvanOers_code_documentatie/sandbox/';
NRINC = 3000; % number of time increments
NEX = 300; NEY = NEX; % grid size
NEX2=2*NEX; NEY2=2*NEY;


hf = figure; set(hf, 'position', [100 100 700 700]);

iteration = 0;
for incr = 0:5:NRINC-1

    ctags = load([path 'ctags' num2str(incr) '.out']);
    bigctags = zeros(NEY2,NEX2); 
    for ey=1:NEY
        for ex=1:NEX
            e = ex+(ey-1)*NEX; 
            tagval = ctags(e);
            bigctags(ey*2-1,ex*2-1)=tagval;
            bigctags(ey*2-1,ex*2  )=tagval;
            bigctags(ey*2  ,ex*2-1)=tagval;
            bigctags(ey*2  ,ex*2  )=tagval;
        end
    end
    
    
    bigfield = 64*ones(NEY2,NEX2);
    for ey=2:NEY2
        for ex=2:NEX2
            % color cells red
            if(bigctags(ey,ex))
                bigfield(ey,ex)=62; 
            end
            % draw cell borders black
            if (bigctags(ey,ex)~=bigctags(ey-1,ex))|(bigctags(ey,ex)~=bigctags(ey,ex-1))
                bigfield(ey,ex)=61;
            end
        end
    end
    
    % make figure
    im = image(bigfield);
    colormap(jet2)
    axis image;
    axis xy;
    axis off;
    
    hold on
    
    
    str_save = load([path 'pstrain' num2str(incr) '.out']);
    str = str_save(:,1)*0.000001; % values of max. princ. strain
    REF = .1; strsc = str/REF; %strains are scaled to REF 
    AMPL = 6; strsc = strsc*AMPL; 
    % strainsvectors of size REF should be one pixel long if AMPL = 1
    SPACING = 3; % show strains every # elements
    SHOWTR = .03; % don't show strains below this size
    
    i = 1; xpos=[]; ypos =[]; strx=[]; stry=[];
    for ey=1:SPACING:NEY
        for ex=1:SPACING:NEX
            e = ex+(ey-1)*NEX;
            
            if(str(e)>SHOWTR)
                xpos(i)=ex*2;
                ypos(i)=ey*2;
                strx(i) = strsc(e)*str_save(e,2)/1000;
                stry(i) = strsc(e)*str_save(e,3)/1000;
                i=i+1;
            end
        end
    end
    
    % add arrows to figure
    q=quiver(xpos,ypos, strx, stry,0); set(q,'Color',[0 0 0],'ShowArrowHead','off');
    q=quiver(xpos,ypos,-strx,-stry,0); set(q,'Color',[0 0 0],'ShowArrowHead','off');
    
    hold off
    drawnow

    % make screenshot of figure (which should remain on forground)
    iteration = iteration+1; M(iteration) = getframe;
end
% join screenshots into avi video
movie2avi(M,'simmovie','fps',25,'compression','Cinepak','quality',100);





