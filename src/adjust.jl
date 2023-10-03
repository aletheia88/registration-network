using Random
using Images
using ImageFiltering

function adjust_img(img, target_dim)

    img_adj = zeros(UInt16, target_dim)
    dim_x, dim_y, dim_z = size(img)
    center_x, center_y, center_z = round.(Int, [dim_x, dim_y, dim_z] / 2)

    dx, dy, dz = round.(Int, target_dim ./ 2)

    rg_x = max(center_x - dx + 1, 1):min(center_x + dx, dim_x)
    rg_y = max(center_y - dy + 1, 1):min(center_y + dy, dim_y)
    rg_z = max(center_z - dz + 1, 1):min(center_z + dz, dim_z)

    rg_x_ = round(Int, dx - length(rg_x) / 2 + 1):round(Int, dx + length(rg_x) / 2)
    rg_y_ = round(Int, dy - length(rg_y) / 2 + 1):round(Int, dy + length(rg_y) / 2)
    rg_z_ = round(Int, dz - length(rg_z) / 2 + 1):round(Int, dz + length(rg_z) / 2)

    img_adj[rg_x_, rg_y_, rg_z_] .= img[rg_x, rg_y, rg_z]
    return img_adj
end

function adjust_image_cm(image, center, target_dim)

    img_adj = zeros(UInt16, target_dim)
    dim_x, dim_y, dim_z = size(image)
    center_x, center_y, center_z = center
    dx, dy, dz = round.(Int, target_dim ./ 2)

    rg_x = max(center_x - dx + 1, 1):min(center_x + dx, dim_x)
    rg_y = max(center_y - dy + 1, 1):min(center_y + dy, dim_y)
    rg_z = max(center_z - dz + 1, 1):min(center_z + dz, dim_z)

    rg_x_ = round(Int, dx - length(rg_x) / 2 + 1):round(Int, dx + length(rg_x) / 2)
    rg_y_ = round(Int, dy - length(rg_y) / 2 + 1):round(Int, dy + length(rg_y) / 2)
    rg_z_ = round(Int, dz - length(rg_z) / 2 + 1):round(Int, dz + length(rg_z) / 2)

    img_adj[rg_x_, rg_y_, rg_z_] .= image[rg_x, rg_y, rg_z]
    return img_adj
end

function adjust_point_v0(image_dim, point, center, target_dim)
    # Extract x, y,  coordinates of the point
    pt_x, pt_y = point
    # Calculate dimensions similar to the adjust_image_cm function
    dim_x, dim_y, _ = image_dim
    center_x, center_y, _ = center
    dx, dy, _ = round.(Int, target_dim ./ 2)

    rg_x = max(center_x - dx + 1, 1):min(center_x + dx, dim_x)
    rg_y = max(center_y - dy + 1, 1):min(center_y + dy, dim_y)

    # Check if the point is within the cropped region
    if pt_x in rg_x && pt_y in rg_y
        # Calculate the new coordinates in the adjusted image
        new_x = pt_x - first(rg_x) + 1
        new_y = pt_y - first(rg_y) + 1
        return (new_x, new_y)
    else
        # Point is outside the cropped region
        return point
    end
end

function adjust_point(image_dim, point, center, target_dim)
    # Extract x, y, coordinates of the point
    pt_x, pt_y = point

    # Calculate dimensions similar to the adjust_image_cm function
    dim_x, dim_y, _ = image_dim
    center_x, center_y, _ = center
    dx, dy, _ = round.(Int, target_dim ./ 2)

    rg_x = max(center_x - dx + 1, 1):min(center_x + dx, dim_x)
    rg_y = max(center_y - dy + 1, 1):min(center_y + dy, dim_y)

    rg_x_ = round(Int, dx - length(rg_x) / 2 + 1):round(Int, dx + length(rg_x) / 2)
    rg_y_ = round(Int, dy - length(rg_y) / 2 + 1):round(Int, dy + length(rg_y) / 2)

    # Check if the point is within the cropped region
    if pt_x in rg_x && pt_y in rg_y
        idx_x = findfirst(==(pt_x), rg_x)
        idx_y = findfirst(==(pt_y), rg_y)
        # Use the index to get the new x and y coordinates from rg_x_ and rg_y_
        new_x = rg_x_[idx_x]
        new_y = rg_y_[idx_y]
        return new_x, new_y
    else
        # Point is outside the cropped region
        return point
    end
end
