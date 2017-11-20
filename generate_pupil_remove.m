%%
dir = 'scenes/'
dir2 = 'patches/'

file1 = {'scene045a.JPG', 'scene046a.JPG', 'scene047a.JPG', 'scene048a.JPG', 'scene048d.JPG', 'scene049a.JPG', 'scene049b.JPG' }
file2 = {'scene045c.JPG', 'scene046c.JPG', 'scene047c.JPG', 'scene048c.JPG', 'scene048c.JPG', 'scene049d.JPG', 'scene049d.JPG' }

n = 0;

for i=1:7
    % eye images
    fname = sprintf('%s/%s',dir, file1{i});
    im1 = imread(fname);  
    % scene images
    fname = sprintf('%s/%s',dir, file2{i});
    im2 = imread(fname);  
    
    % get patches
    for j=1:50
        x = randi(size(im1,2));
        y = randi(size(im1,1));
        wx = randi(size(im1,2)-x);
        wy = randi(size(im1,1)-y);
        patch1 = im1(y:(y+wy),x:(x+wx),:);
        patch2 = im2(y:(y+wy),x:(x+wx),:);
        
        sprintf('%s/%s/%05d.jpg',dir2,'src',n)
        
        imwrite(patch1,sprintf('%s/%s/%05d.jpg',dir2,'src',n));
        imwrite(patch2,sprintf('%s/%s/%05d.jpg',dir2,'dst',n));
        n = n + 1;
    end
end

