import os
import xml.etree.ElementTree as ET
import numpy as np
import torch


def construct_urdf(
    link_idx,
    link_name,
    origin_xyz,
    axis_xyz,
    obj_path,
    urdf_path,
    lower_upper_limits=None,
    motion_type=None,
):
    if link_idx == 0:
        root = ET.Element("robot", name="generated_robot")
        tree = ET.ElementTree(root)
        base_link = ET.SubElement(root, "link", name="base")
        link_elem = ET.SubElement(root, "link", name=link_name)
        visual = ET.SubElement(link_elem, "visual", name=link_name)
        ET.SubElement(visual, "origin", xyz="0 0 0")
        visual_geom = ET.SubElement(visual, "geometry")
        ET.SubElement(visual_geom, "mesh", filename=obj_path)
        collision = ET.SubElement(link_elem, "collision", name=link_name)
        ET.SubElement(collision, "origin", xyz="0 0 0")
        collision_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geom, "mesh", filename=obj_path)
        joint_elem = ET.SubElement(
            root, "joint", name=f"joint_{link_idx}", type="fixed"
        )
        ET.SubElement(joint_elem, "origin", rpy="1.5708 0 -1.5708", xyz="0 0 0")
        ET.SubElement(joint_elem, "child", link=link_name)
        ET.SubElement(joint_elem, "parent", link="base")
    else:
        if not os.path.exists(urdf_path):
            raise ValueError(
                f"URDF file does not exist: {urdf_path}, please create link_0 first"
            )
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        link_elem = ET.SubElement(root, "link", name=link_name)
        if origin_xyz is not None:
            if isinstance(origin_xyz, torch.Tensor):
                origin_xyz = origin_xyz.cpu().numpy()
            origin_str = f"{origin_xyz[0]:.6f} {origin_xyz[1]:.6f} {origin_xyz[2]:.6f}"
            neg_origin_str = f"{-origin_xyz[0]:.6f} {-origin_xyz[1]:.6f} {-origin_xyz[2]:.6f}"
        else:
            origin_str = "0 0 0"
            neg_origin_str = "0 0 0"
        if axis_xyz is not None:
            if isinstance(axis_xyz, torch.Tensor):
                axis_xyz = axis_xyz.cpu().numpy()
            axis_xyz = np.asarray(axis_xyz, dtype=np.float64)
            axis_norm = np.linalg.norm(axis_xyz)
            if np.isfinite(axis_norm) and axis_norm > 1e-8:
                axis_xyz = axis_xyz / axis_norm
                axis_str = " ".join(f"{value:.6f}" for value in axis_xyz)
            else:
                axis_str = "1 0 0"
        else:
            axis_str = "1 0 0"
        visual = ET.SubElement(link_elem, "visual", name=link_name)
        ET.SubElement(visual, "origin", xyz=neg_origin_str)
        visual_geom = ET.SubElement(visual, "geometry")
        ET.SubElement(visual_geom, "mesh", filename=obj_path)
        collision = ET.SubElement(link_elem, "collision", name=link_name)
        ET.SubElement(collision, "origin", xyz=neg_origin_str)
        collision_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geom, "mesh", filename=obj_path)
        if motion_type is not None:
            if isinstance(motion_type, torch.Tensor):
                if motion_type.dim() > 0 and motion_type.numel() > 1:
                    motion_type_val = torch.argmax(motion_type.cpu()).item()
                else:
                    motion_type_val = motion_type.cpu().item()
                joint_type = "revolute" if motion_type_val == 0 else "prismatic"
            elif isinstance(motion_type, str):
                joint_type = motion_type
            else:
                joint_type = "revolute" if motion_type == 0 else "prismatic"
        else:
            joint_type = "revolute"
        joint_elem = ET.SubElement(
            root, "joint", name=f"joint_{link_idx}", type=joint_type
        )
        ET.SubElement(joint_elem, "origin", xyz=origin_str)
        ET.SubElement(joint_elem, "axis", xyz=axis_str)
        ET.SubElement(joint_elem, "child", link=link_name)
        ET.SubElement(joint_elem, "parent", link="link_0")
        limit_elem = ET.SubElement(joint_elem, "limit")
        if lower_upper_limits is not None:
            if isinstance(lower_upper_limits, torch.Tensor):
                lower_upper_limits = lower_upper_limits.cpu().numpy()
            lower_upper_limits = np.asarray(
                lower_upper_limits, dtype=np.float64
            ).reshape(-1)
            if lower_upper_limits.size >= 2 and np.isfinite(
                lower_upper_limits[:2]
            ).all():
                lower, upper = np.sort(lower_upper_limits[:2])
                limit_elem.set("lower", f"{lower:.6f}")
                limit_elem.set("upper", f"{upper:.6f}")
            else:
                limit_elem.set("lower", "-3.14159")
                limit_elem.set("upper", "3.14159")
        else:
            limit_elem.set("lower", "-3.14159")
            limit_elem.set("upper", "3.14159")
    ET.indent(tree, space="    ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"URDF file saved: {urdf_path}")
