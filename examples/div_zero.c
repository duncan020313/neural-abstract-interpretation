main(n)
{
  x = 10;
  y = n;
  z = 0;
  if (y == 0)
  {
    z = 1;
  }
  else
  {
    z = 2;
  }
  x = x / y;
  y %= z;
  return x;
}
