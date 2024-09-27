mod sampler;

use sampler::State;

fn main() {
    const N_STEP: usize = 1000;
    const N_DIM: usize = 2;
    let mut state = State::<N_DIM>::new(0xdeadbeef);
    for _ in 0..N_STEP {
        state.take_step();
        println!("{:?}, {:?}", state.arr[0], state.arr[1]);
    }
}
