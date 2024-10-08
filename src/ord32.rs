//! f32 stored as i32 to make it comparable and faster to compare.

/// Ord32 is a wrapper around i32 that implements Ord and PartialEq.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Ord32(i32);

/// TODO: mark as const fn when it's stable.
impl Ord32 {
    /// Create a new Ord32 from an f32.
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits() as i32;
        let mask = ((bits >> 31) as u32) >> 1;
        let res = bits ^ (mask as i32);
        Self(res)
    }

    /// Convert Ord32 to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let mask = ((bits >> 31) as u32) >> 1;
        let res = bits ^ (mask as i32);
        f32::from_bits(res as u32)
    }
}

impl From<f32> for Ord32 {
    #[inline]
    fn from(x: f32) -> Self {
        Self::from_f32(x)
    }
}

impl From<Ord32> for f32 {
    #[inline]
    fn from(x: Ord32) -> Self {
        x.to_f32()
    }
}

/// AlwaysEqual is a wrapper around T that always returns true for PartialEq.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct AlwaysEqual<T>(pub T);

impl<T> PartialEq for AlwaysEqual<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> Eq for AlwaysEqual<T> {}

impl<T> PartialOrd for AlwaysEqual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for AlwaysEqual<T> {
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}
