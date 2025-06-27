import mujoco

def main(): 

    path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/data/mimicgen/model.xml'
    model = mujoco.MjModel.from_xml_path(path)

    # Get the number of DOFs
    nq = model.nq
    nv = model.nv

    # Get names for qpos/qvel entries
    qpos_names = [model.names[i] for i in model.joint_qposadr]
    qvel_names = [model.names[i] for i in model.joint_dofadr]

if __name__ == '__main__': 
    main()
    